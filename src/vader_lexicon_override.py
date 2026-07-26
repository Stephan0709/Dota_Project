"""
Domain-specific VADER lexicon override for Dota 2 all-chat (section 4.5.2).

This is the single source of truth for the override - imported both by the
validation-study re-check (validate_lexicon_override.py) and, later, by the
full-dataset trajectory scoring pipeline (Task 5+), so the exact same
correction is applied everywhere rather than being re-typed in multiple
places and risking drift.

Each entry's justification is based on direct inspection of the 200-message
human-annotated validation sample (see data/interim/validation_sample.csv):

- "ez": already in VADER's default lexicon at +1.5 (inherited from generic
  "easy"). In this sample, 15/15 messages containing "ez" were human-labelled
  negative - it functions as a sarcastic taunt ("that was easy [for us]",
  mocking the opponent), not a genuine positive judgement. Overridden to -1.5.

- "feed": not in VADER's default lexicon at all (contributes 0). In this
  domain "feeding" specifically means playing badly / repeatedly dying,
  always used critically. Small sample (n=2) but consistent direction and
  well-documented domain meaning. Overridden to -1.8.

- "throw": not in VADER's default lexicon. "Throwing" means wasting a
  winning position - always negative in context. Only n=1 in the validation
  sample, weaker direct evidence, but consistent with well-known Dota usage.
  Overridden to -1.0 (moderate, reflecting the thinner evidence base).

- "win" / "won": VADER's default scores (+2.8 / +2.7) are appropriate for
  general English but too strong here - most non-"ez" instances in the
  sample are neutral factual statements about match state ("won the game",
  "auto win"), with one genuine positive use ("win plz"). Dampened to +0.4
  rather than fully zeroed, to avoid eliminating genuine positive usage
  entirely.

Terms checked but NOT overridden, since they already perform reasonably
against the validation sample:
- "gg" (+1.2 in VADER): 25/29 (86%) agreement with human labels - mostly
  genuine "good game", left unchanged.
- "noob" (-0.2 in VADER): 4/5 (80%) agreement - left unchanged.

Empirical validation (see validate_lexicon_override.py):
    Baseline VADER (no override):      accuracy=0.615  macro F1=0.605
    With full override applied:        accuracy=0.695  macro F1=0.680
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

LEXICON_OVERRIDE = {
    "ez": -1.5,
    "feed": -1.8,
    "throw": -1.0,
    "win": 0.4,
    "won": 0.4,
}


def get_analyzer_with_override():
    """Return a VADER SentimentIntensityAnalyzer with the domain override
    applied. Use this everywhere VADER scoring happens from now on, instead
    of instantiating a bare SentimentIntensityAnalyzer()."""
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(LEXICON_OVERRIDE)
    return analyzer
