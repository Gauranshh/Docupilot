"""A small machine-learning model, trained on labelled questions, that routes questions.

Why: the LLM router costs one API call per question (free tiers allow ~20 calls/day).
A TF-IDF + logistic-regression classifier answers in about a millisecond, offline, and
handles most questions; the LLM router is only asked when the model is unsure.

    python -m docupilot.cli train-router          # trains, evaluates, saves models/router.joblib

Labels: retrieve (a specific fact), document (whole-document: summaries, counts, overviews),
direct (greetings, thanks, questions about the assistant).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("retrieve", "document", "direct")
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "router_train.csv"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "router.joblib"


def load_dataset(path: Path = DEFAULT_DATA) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with Path(path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label, text = row["label"].strip(), row["question"].strip()
            if label not in LABELS:
                raise ValueError(f"Unknown label {label!r} in {path}")
            if text:
                texts.append(text)
                labels.append(label)
    return texts, labels


def build_pipeline():
    """Word 1–2-grams catch phrases ("how many", "summarize the"); character 2–5-grams
    catch typos and short forms ("summarise", "hii", "tldr")."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, lowercase=True)),
    ])
    clf = LogisticRegression(C=8.0, class_weight="balanced", max_iter=2000)
    return Pipeline([("features", features), ("clf", clf)])


def train(data_path: Path = DEFAULT_DATA, model_path: Path = DEFAULT_MODEL,
          test_size: float = 0.25, seed: int = 42) -> dict:
    """Train with a held-out test split and 5-fold cross-validation, then fit on all data and save."""
    import joblib
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

    texts, labels = load_dataset(data_path)
    x_tr, x_te, y_tr, y_te = train_test_split(texts, labels, test_size=test_size,
                                              stratify=labels, random_state=seed)
    model = build_pipeline().fit(x_tr, y_tr)
    pred = model.predict(x_te)
    cv = cross_val_score(build_pipeline(), texts, labels,
                         cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed))

    # Accuracy when the model is confident: the app only trusts it above `threshold`
    from sklearn.model_selection import cross_val_predict

    probs = cross_val_predict(build_pipeline(), texts, labels, method="predict_proba",
                              cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed))
    classes = sorted(set(labels))
    confident = {}
    for threshold in (0.6, 0.7, 0.8):
        picked = [(classes[int(p.argmax())], y) for p, y in zip(probs, labels) if p.max() >= threshold]
        confident[str(threshold)] = {
            "coverage": round(len(picked) / len(labels), 4),
            "accuracy": round(sum(a == b for a, b in picked) / max(len(picked), 1), 4),
        }

    final = build_pipeline().fit(texts, labels)  # ship the model trained on every example
    metrics = {
        "confident_cv": confident,
        "examples": len(texts),
        "per_label": {lab: labels.count(lab) for lab in LABELS},
        "holdout_size": len(x_te),
        "holdout_accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "cv5_accuracy_mean": round(float(cv.mean()), 4),
        "cv5_accuracy_std": round(float(cv.std()), 4),
        "confusion_matrix": {"labels": list(LABELS),
                             "matrix": confusion_matrix(y_te, pred, labels=list(LABELS)).tolist()},
        "report": classification_report(y_te, pred, labels=list(LABELS), output_dict=True, zero_division=0),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "labels": list(LABELS), "metrics": metrics}, model_path)
    model_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


@dataclass
class TrainedRouter:
    """Loads the saved model; `predict` returns (label, confidence between 0 and 1)."""

    pipeline: object
    labels: list[str]

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL) -> "TrainedRouter | None":
        path = Path(path)
        if not path.exists():
            return None
        try:
            import joblib

            bundle = joblib.load(path)  # our own model file, shipped in the repository
            return cls(pipeline=bundle["pipeline"], labels=bundle["labels"])
        except Exception as e:  # noqa: BLE001 — a broken model must never break the app
            print(f"(trained router unavailable: {type(e).__name__}: {e})")
            return None

    def predict(self, question: str) -> tuple[str, float]:
        probs = self.pipeline.predict_proba([question])[0]
        classes = list(self.pipeline.classes_)
        best = int(probs.argmax())
        return str(classes[best]), float(probs[best])
