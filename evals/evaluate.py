"""Score recorded full-workflow outputs against a human-reviewed JSONL dataset.
Citation validity checks IDs only, not whether prose is factually supported.
"""
import argparse
import json
from pathlib import Path


def evaluate(cases, outputs):
    results = {item['case_id']: item for item in outputs}
    recall, citation_validity, abstention, latencies = [], [], [], []
    for case in cases:
        result = results.get(case['case_id'])
        if result is None:
            raise ValueError(f"Missing output: {case['case_id']}")
        evidence = result.get('evidence', [])
        retrieved_urls = {item['url'] for item in evidence}
        expected = set(case.get('expected_source_urls', []))
        if expected:
            recall.append(len(expected & retrieved_urls) / len(expected))
        ids = result.get('cited_evidence_ids', [])
        allowed = {item['id'] for item in evidence}
        if ids:
            citation_validity.append(sum(i in allowed for i in ids)/len(ids))
        if case.get('must_abstain'):
            abstention.append(result.get('analysis_status') == 'unavailable' and result.get('signal') == 'ERROR')
        if 'latency_ms' in result:
            latencies.append(result['latency_ms'])
    mean = lambda values: sum(values)/len(values) if values else None
    return {'cases':len(cases), 'source_recall':mean(recall), 'citation_id_validity':mean(citation_validity),
            'required_abstention_rate':mean(abstention), 'mean_latency_ms':mean(latencies),
            'note':'Citation IDs and retrieval coverage do not establish factual correctness. Review claims against cited excerpts separately. No trading-return metric is computed.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',required=True)
    parser.add_argument('--outputs',required=True)
    args=parser.parse_args()
    load=lambda file:[json.loads(line) for line in Path(file).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    print(json.dumps(evaluate(load(args.cases),load(args.outputs)),indent=2))
