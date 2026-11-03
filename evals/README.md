# Evaluation workflow

The checked-in cases are failure scenarios for mocked full-workflow runs, not a
financial accuracy benchmark. The pytest suite runs these kinds of failure paths
without a model account. No accuracy, latency or investment-return claim is made.

For a real held-out dataset:
1. Freeze company, market region, as-of timestamp, news and filing snapshots.
2. Have a reviewer label expected source URLs and situations requiring abstention.
3. Record the deployed workflow's outputs with case_id, evidence,
   cited_evidence_ids, signal, analysis_status, and measured latency_ms.
4. Run `python evals/evaluate.py --cases heldout.jsonl --outputs recorded.jsonl`.
5. Review each numerical/factual claim against cited excerpts separately. An ID
   being valid is necessary, but does not prove the claim is supported.
6. Compare identical snapshots before/after changes; keep development and held-out
   cases separate. Compare cached and uncached latency separately.

The old backtest.py uses a different prompt from the deployed workflow and must
not be reported as a measurement of this RAG pipeline. Historical evaluation must
exclude documents/news published after each decision time; execution assumptions,
fees and slippage need an independently specified simulator.
