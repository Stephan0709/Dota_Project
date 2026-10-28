

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"

# Standard VADER thresholds (Hutto & Gilbert 2014) for binarising the
# compound score into three classes.
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

# cardiffnlp/twitter-roberta-base-sentiment label mapping
ROBERTA_LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}

LABELS_ORDER = ["negative", "neutral", "positive"]


def vader_predict(compound):
    if compound >= VADER_POS_THRESHOLD:
        return "positive"
    if compound <= VADER_NEG_THRESHOLD:
        return "negative"
    return "neutral"


def run_roberta(texts):
    from transformers import pipeline
    print("Loading twitter-roberta-base-sentiment (downloads ~500MB on first run)...")
    clf = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        truncation=True,
    )
    print(f"Scoring {len(texts)} messages with RoBERTa...")
    results = clf(list(texts), batch_size=16)
    labels = [ROBERTA_LABEL_MAP.get(r["label"], r["label"]) for r in results]
    scores = [r["score"] for r in results]
    return labels, scores


def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS_ORDER, zero_division=0)
    print(f"\n{'=' * 50}")
    print(f"{name}")
    print(f"{'=' * 50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, labels=LABELS_ORDER, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted), order:", LABELS_ORDER)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_ORDER)
    print(cm)
    return acc, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str,
        default=str(OUTPUT_DIR / "validation_sample.csv"),
    )
    args = parser.parse_args()

    path = Path(args.input)
    df = pd.read_csv(path)

    required_cols = {"text", "human_label"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing required columns: {missing}")
        return

    df["human_label"] = df["human_label"].str.strip().str.lower()

    # --- VADER ---
    if "vader_compound" in df.columns:
        print("Using existing vader_compound column (already computed during sampling).")
        compounds = df["vader_compound"]
    else:
        print("No vader_compound column found - recomputing with VADER.")
        analyzer = SentimentIntensityAnalyzer()
        compounds = df["text"].astype(str).apply(lambda t: analyzer.polarity_scores(t)["compound"])
    df["vader_pred"] = compounds.apply(vader_predict)

    # --- RoBERTa ---
    roberta_labels, roberta_scores = run_roberta(df["text"].astype(str).tolist())
    df["roberta_pred"] = roberta_labels
    df["roberta_confidence"] = roberta_scores

    # --- Metrics ---
    vader_acc, vader_f1 = print_metrics("VADER", df["human_label"], df["vader_pred"])
    roberta_acc, roberta_f1 = print_metrics("twitter-roberta-base-sentiment", df["human_label"], df["roberta_pred"])

    # --- Scorer selection per proposal section 4.5.2 ---
    print(f"\n{'=' * 50}")
    print("SCORER SELECTION")
    print(f"{'=' * 50}")
    print(f"VADER macro F1:   {vader_f1:.4f}")
    print(f"RoBERTa macro F1: {roberta_f1:.4f}")
    diff = abs(vader_f1 - roberta_f1)
    print(f"Difference:       {diff:.4f}")

    if diff <= 0.05:
        primary = "VADER"
        reason = "within 0.05 of each other - VADER preferred for interpretability and continuous score"
    elif roberta_f1 > vader_f1:
        primary = "RoBERTa"
        reason = "RoBERTa macro F1 substantially higher"
    else:
        primary = "VADER"
        reason = "VADER macro F1 substantially higher"
    print(f"\n>>> Primary scorer: {primary} ({reason})")

    # --- Disagreement analysis: cases where each scorer got it wrong ---
    df["vader_correct"] = df["vader_pred"] == df["human_label"]
    df["roberta_correct"] = df["roberta_pred"] == df["human_label"]

    vader_wrong_roberta_right = df[(~df["vader_correct"]) & (df["roberta_correct"])]
    roberta_wrong_vader_right = df[(~df["roberta_correct"]) & (df["vader_correct"])]

    print(f"\n{'=' * 50}")
    print("DISAGREEMENT ANALYSIS")
    print(f"{'=' * 50}")
    print(f"VADER wrong, RoBERTa right: {len(vader_wrong_roberta_right)} messages")
    print(f"RoBERTa wrong, VADER right: {len(roberta_wrong_vader_right)} messages")

    if len(vader_wrong_roberta_right) > 0:
        print("\nSample messages where VADER failed but RoBERTa succeeded:")
        print(vader_wrong_roberta_right[["text", "human_label", "vader_pred", "roberta_pred"]].head(15).to_string(index=False))

    if len(roberta_wrong_vader_right) > 0:
        print("\nSample messages where RoBERTa failed but VADER succeeded:")
        print(roberta_wrong_vader_right[["text", "human_label", "vader_pred", "roberta_pred"]].head(15).to_string(index=False))

    # --- Save full results ---
    out_path = OUTPUT_DIR / "scorer_comparison_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results (all predictions, correctness flags) saved to: {out_path}")


if __name__ == "__main__":
    main()
