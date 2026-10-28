

from pathlib import Path

import pandas as pd

INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"

TYPES = {
    "closing_gg_wp":          [104, 164, 103],
    "gloating_ez":            [114, 105, 115, 14, 112, 140],
    "surrender_end_requests": [88, 150],
    "outcome_statements":     [138, 139],
    "laughter_emotes":        [163, 159, 157, 160, 165, 168, 162, 123, 90],
    "short_answers_filler":   [152, 169, 153, 166, 146, 76, 145, 167, 144, 143, 141, 113, 136, 142, 92],
    "evaluation_praise":      [122, 56, 62, 98, 33, 27, 82, 133, 17],
    "hostility_insults":      [51, 130, 131, 94, 135, 69, 36, 118, 119, 53, 134],
    "blame_frustration":      [125, 137, 8, 54, 89, 42, 48],
    "report_commend":         [66, 29],
    "game_talk_callouts":     [59, 86, 91, 147, 4, 149, 28, 52, 40, 10, 61, 41, 49, 132, 120, 15, 30, 50, 71, 58, 85, 38, 35, 5, 108, 107, 39],
    "hero_name_mentions":     [67, 11, 45, 24, 9, 37, 3, 16, 18, 2, 12, 22, 60, 21, 84, 32, 23, 116, 25, 19, 20, 26, 46, 44, 6, 64, 34, 43, 13, 1, 47],
    "numbers_scores":         [55, 68, 78, 73, 77],
    "tech_afk_lag":           [65, 97, 7, 121, 96],
    "nationality_banter":     [72, 79, 80, 75],
    "non_english":            [171, 102, 175, 31, 83, 101, 129, 93, 124, 148, 117, 154, 174, 127, 128, 155, 172, 173, 81, 170, 156, 151, 158, 126, 99],
    "other_unclear":          [74, 0, 100, 161, 95, 106, 111, 70, 57, 109, 110, 63, 87],
}

FAMILY = {
    "closing_gg_wp": "end_of_game", "gloating_ez": "end_of_game",
    "surrender_end_requests": "end_of_game", "outcome_statements": "end_of_game",
    "laughter_emotes": "social", "short_answers_filler": "social", "evaluation_praise": "social",
    "hostility_insults": "hostile", "blame_frustration": "hostile",
    "report_commend": "moderation",
    "game_talk_callouts": "game_content", "hero_name_mentions": "game_content",
    "numbers_scores": "game_content", "tech_afk_lag": "game_content",
    "nationality_banter": "other", "non_english": "other", "other_unclear": "other",
}

# The 21 clusters flagged as judgement calls were reviewed by the researcher (all proposed changes accepted).
REVIEWED = {164, 133, 125, 137, 74, 108, 17, 107, 0, 99, 28, 16, 41, 106, 95, 119, 90, 15, 140, 39, 134}
# clusters whose type CHANGED during review -> original (proposed) type, kept for the audit trail
MOVED_FROM = {133: "closing_gg_wp", 108: "blame_frustration", 17: "other_unclear", 107: "other_unclear",
              99: "laughter_emotes", 90: "other_unclear", 39: "gloating_ez"}


def main():
    s = pd.read_csv(INTERIM_DIR / "msgclust_cluster_summary.csv")
    cl2type = {c: t for t, ids in TYPES.items() for c in ids}
    dup = [c for c in cl2type if sum(c in ids for ids in TYPES.values()) > 1]
    assert not dup, f"clusters assigned twice: {dup}"

    s["type"] = s["cluster"].map(cl2type)
    unmapped = s[s["type"].isna()]
    if len(unmapped):
        print(f"WARNING: {len(unmapped)} clusters not in the mapping -> 'other_unclear' (basis=unreviewed):")
        for _, r in unmapped.iterrows():
            print(f"   c{int(r.cluster)} {r.share_of_all_messages:.2%} | {r.top_messages[:80]}")
    s["basis"] = s["cluster"].map(lambda c: "unreviewed" if c not in cl2type else ("reviewed" if c in REVIEWED else "llm_proposed"))
    s["moved_in_review_from"] = s["cluster"].map(MOVED_FROM).fillna("")
    s["type"] = s["type"].fillna("other_unclear")
    s["family"] = s["type"].map(FAMILY)
    extra = set(cl2type) - set(s["cluster"])
    if extra:
        print("note: ids in mapping but not in this summary:", sorted(extra))

    out = s[["cluster", "type", "family", "basis", "moved_in_review_from", "share_of_all_messages", "top_messages"]].sort_values(
        ["family", "type", "share_of_all_messages"], ascending=[True, True, False])
    out["top_messages"] = out["top_messages"].str.slice(0, 80)
    out.to_csv(INTERIM_DIR / "cluster_type_map.csv", index=False)

    print("\nShare of ALL messages by type (clustered messages only; HDBSCAN noise is separate):")
    g = s.groupby(["family", "type"])["share_of_all_messages"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    for (fam, t), r in g.iterrows():
        print(f"  {fam:13s} {t:24s} {r['sum']:6.2%}  ({int(r['count'])} clusters)")
    print(f"  total clustered: {s['share_of_all_messages'].sum():.2%}")
    print("\nbasis counts:", s["basis"].value_counts().to_dict())


if __name__ == "__main__":
    main()
