"""Rank raw similarities and compute source-overlap hits."""

import math


def rank(query, scores):
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError(f"Non-finite similarity for {query['query_id']}")
    relevant = set(query["relevant_ids"])
    if len(scores) < 3 or not relevant or not relevant.issubset(scores):
        raise ValueError(f"Incomplete gallery for {query['query_id']}")
    order = sorted(scores, key=lambda candidate: (-scores[candidate], candidate))
    hit = bool(relevant.intersection(order[:3]))
    return [
        dict(query_id=query["query_id"], source_id=query["source_id"], candidate_id=candidate,
             rank=position, similarity_score=scores[candidate], relevant=candidate in relevant,
             top3_success=hit)
        for position, candidate in enumerate(order, start=1)
    ]


def hit_metrics(rankings):
    first = [row for row in rankings if row["rank"] == 1]
    total = len(first)
    if not total:
        raise ValueError("No evaluated queries")
    successful = sum(row["top3_success"] for row in first)
    return dict(successful_queries=successful, total_queries=total, top3_hit_rate=successful / total)
