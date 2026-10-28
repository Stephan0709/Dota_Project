

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from vader_lexicon_override import LEXICON_OVERRIDE, get_analyzer_with_override

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"
LABELS_ORDER = ["negative", "neutral", "positive"]


def classify(compound):
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(OUTPUT_DIR / "validation_sample.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["human_label"] = df["human_label"].str.strip().str.lower()

    baseline_analyzer = SentimentIntensityAnalyzer()
    override_analyzer = get_analyzer_with_override()

    df["vader_baseline_compound"] = df["text"].astype(str).apply(
        lambda t: baseline_analyzer.polarity_scores(t)["compound"]
    )
    df["vader_override_compound"] = df["text"].astype(str).apply(
        lambda t: override_analyzer.polarity_scores(t)["compound"]
    )
    df["vader_baseline_pred"] = df["vader_baseline_compound"].apply(classify)
    df["vader_override_pred"] = df["vader_override_compound"].apply(classify)

    print("Lexicon override applied:")
    for term, score in LEXICON_OVERRIDE.items():
        print(f"  {term!r}: -> {score}")

    def report(name, preds):
        acc = accuracy_score(df["human_label"], preds)
        f1 = f1_score(df["human_label"], preds, average="macro", labels=LABELS_ORDER, zero_division=0)
        print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Macro F1:  {f1:.4f}")
        print(classification_report(df["human_label"], preds, labels=LABELS_ORDER, zero_division=0))
        print("Confusion matrix (rows=true, cols=predicted), order:", LABELS_ORDER)
        print(confusion_matrix(df["human_label"], preds, labels=LABELS_ORDER))
        return acc, f1

    base_acc, base_f1 = report("BASELINE VADER (no override)", df["vader_baseline_pred"])
    over_acc, over_f1 = report("VADER WITH DOMAIN LEXICON OVERRIDE", df["vader_override_pred"])

    print(f"\n{'=' * 50}\nSUMMARY\n{'=' * 50}")
    print(f"Accuracy:  {base_acc:.4f} -> {over_acc:.4f}  (change: {over_acc - base_acc:+.4f})")
    print(f"Macro F1:  {base_f1:.4f} -> {over_f1:.4f}  (change: {over_f1 - base_f1:+.4f})")

    # Show exactly which messages changed prediction, and whether that was
    # an improvement or a regression - important for an honest write-up.
    changed = df[df["vader_baseline_pred"] != df["vader_override_pred"]].copy()
    changed["fixed"] = (
        (changed["vader_baseline_pred"] != changed["human_label"])
        & (changed["vader_override_pred"] == changed["human_label"])
    )
    changed["broken"] = (
        (changed["vader_baseline_pred"] == changed["human_label"])
        & (changed["vader_override_pred"] != changed["human_label"])
    )
    print(f"\nMessages whose prediction changed: {len(changed)}")
    print(f"  Fixed by override (was wrong, now right): {changed['fixed'].sum()}")
    print(f"  Broken by override (was right, now wrong): {changed['broken'].sum()}")

    if changed["broken"].sum() > 0:
        print("\nMessages the override made WORSE (worth reviewing):")
        print(changed[changed["broken"]][
            ["text", "human_label", "vader_baseline_pred", "vader_override_pred"]
        ].to_string(index=False))

    out_path = OUTPUT_DIR / "validation_sample_with_override.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
