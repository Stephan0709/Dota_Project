"""
Environment verification script.

Run this after `pip install -r requirements.txt` to confirm every
dependency for the sentiment trajectory pipeline is correctly installed
and working, BEFORE spending time on real data collection or annotation.

Usage:
    python verify_environment.py
"""

import sys


def check(label, fn):
    try:
        fn()
        print(f"[OK]   {label}")
        return True
    except Exception as e:
        print(f"[FAIL] {label} -> {e}")
        return False


def check_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores("gg ez that was a great teamfight")
    assert "compound" in scores
    print(f"       sample compound score: {scores['compound']}")


def check_transformers_roberta():
    # This downloads ~500MB on first run - only do this once you're ready.
    from transformers import pipeline
    clf = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
    )
    result = clf("gg ez that was a great teamfight")
    print(f"       sample RoBERTa result: {result}")


def check_scipy_stats():
    from scipy.stats import mannwhitneyu, spearmanr
    a = [1, 2, 3, 4, 5]
    b = [2, 3, 4, 5, 6]
    mannwhitneyu(a, b)
    spearmanr(a, b)


def check_sklearn_metrics():
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    y_true = [0, 1, 2, 1, 0]
    y_pred = [0, 1, 1, 1, 0]
    f1_score(y_true, y_pred, average="macro")
    accuracy_score(y_true, y_pred)
    confusion_matrix(y_true, y_pred)


def check_ruptures():
    import numpy as np
    import ruptures as rpt
    signal = np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)])
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=10)
    print(f"       sample changepoints: {result}")


def check_pandas_numpy():
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({"a": np.arange(5)})
    assert len(df) == 5


def check_dotenv():
    from dotenv import load_dotenv
    import os
    load_dotenv()
    key = os.getenv("OPENDOTA_API_KEY")
    if key:
        print("       OPENDOTA_API_KEY found in .env")
    else:
        print("       WARNING: OPENDOTA_API_KEY not set in .env yet")


def check_opendota_connectivity():
    import requests
    # /api/heroes is a lightweight, no-auth-required endpoint - good for
    # a pure connectivity smoke test before doing real match pulls.
    resp = requests.get("https://api.opendota.com/api/heroes", timeout=10)
    resp.raise_for_status()
    heroes = resp.json()
    assert isinstance(heroes, list) and len(heroes) > 0
    print(f"       reached OpenDota API, {len(heroes)} heroes returned")


if __name__ == "__main__":
    print(f"Python version: {sys.version}\n")

    results = []
    results.append(check("pandas / numpy", check_pandas_numpy))
    results.append(check("VADER sentiment scorer", check_vader))
    results.append(check("scipy stats (Mann-Whitney U, Spearman)", check_scipy_stats))
    results.append(check("scikit-learn metrics", check_sklearn_metrics))
    results.append(check("ruptures (PELT change-point detection)", check_ruptures))
    results.append(check("python-dotenv / .env loading", check_dotenv))
    results.append(check("OpenDota API connectivity", check_opendota_connectivity))
    results.append(
        check(
            "transformers + twitter-roberta-base-sentiment (downloads ~500MB first run)",
            check_transformers_roberta,
        )
    )

    print("\n" + "=" * 50)
    passed = sum(results)
    print(f"{passed}/{len(results)} checks passed")
    if passed < len(results):
        print("Fix the FAILs above before starting Task 2 (validation study).")
