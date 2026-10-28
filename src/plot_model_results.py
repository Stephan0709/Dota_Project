

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"
ORDER = ["10min", "15min", "20min", "30min", "full"]
LABEL = {"10min": "10 min", "15min": "15 min", "20min": "20 min", "30min": "30 min", "full": "Full match"}

SERIES = [  # (feature_set, legend label, colour)
    ("sent+types", "Sentiment + message types (all)", "#E8A33D"),
    ("sent+types_noend", "Sentiment + message types, no end-of-game types", "#1C7293"),
    ("sentiment", "Sentiment only (VADER slope/volatility/mean)", "#1E2761"),
    ("volume", "Chat volume only (control)", "#9CA3AF"),
]


def main():
    r = pd.read_csv(INTERIM_DIR / "model_results.csv")
    rf = r[r["model"] == "forest"]
    n = rf.drop_duplicates("cutoff").set_index("cutoff")
    x = range(len(ORDER))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.plot(x, [n.loc[c, "baseline"] for c in ORDER], color="#6B7280", ls="--", lw=1.5, label="Baseline (always pick the more common winner)")
    for sname, label, colour in SERIES:
        d = rf[rf["feature_set"] == sname].set_index("cutoff").loc[ORDER]
        ax.errorbar(x, d["accuracy"], yerr=d["acc_ci95"], color=colour, lw=2.2, marker="o", ms=6, capsize=3, label=label)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{LABEL[c]}\n{int(n.loc[c, 'n_matches']):,} matches" for c in ORDER], fontsize=10)
    ax.set_ylabel("Accuracy (5-fold cross-validated, 95% CI)", fontsize=11)
    ax.set_title("Predicting the winner from team chat: accuracy by how much of the match is seen",
                 fontsize=13, fontweight="bold", color="#1E2761", pad=14)
    ax.set_ylim(0.49, 0.70)
    ax.grid(axis="y", color="#E5E7EB")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=9.5, frameon=False)
    plt.tight_layout()
    out = INTERIM_DIR / "model_accuracy_by_cutoff.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
