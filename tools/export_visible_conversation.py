#!/usr/bin/env python3
"""Preserve a supplied genuine Codex JSONL; extract visible user/assistant messages.

No invented content, summaries, hidden reasoning, system or developer messages are
included in the Markdown extraction. Tool events remain in the local raw backup.
This format is based on the observed local session and may evolve with Codex.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Explicit current-session rollout JSONL, not a guessed latest file')
    parser.add_argument('--output-dir', type=Path, default=Path('docs/ai_logs'))
    args = parser.parse_args()
    raw = args.source.read_bytes()
    events = []
    for line in raw.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            raise ValueError('Source contains an incomplete JSONL line; retry after the current write')
    meta = next(e['payload'] for e in events if e.get('type') == 'session_meta')
    timestamp = datetime.now(timezone.utc).isoformat()
    parts = ['# Genuine Codex visible-message extraction', '',
             'Source session: `' + meta['id'] + '`.', '',
             'Snapshot taken at ' + timestamp + '. This in-progress snapshot ends at the last message saved below; it does not include later messages.', '',
             'Markdown scope: user and assistant message records only. Tool calls/results remain in the raw JSONL backup. Hidden reasoning, system and developer records are excluded from this Markdown file. This is an extraction, not a reconstructed conversation.', '']
    count = 0
    for event in events:
        if event.get('type') != 'response_item':
            continue
        item = event.get('payload', {})
        if item.get('type') != 'message' or item.get('role') not in ('user', 'assistant'):
            continue
        if item.get('channel') not in (None, 'commentary', 'final'):
            continue
        messages = [c['text'] for c in item.get('content', [])
                    if isinstance(c, dict) and c.get('type') in ('input_text', 'output_text', 'text') and 'text' in c]
        if messages:
            count += 1
            parts.extend(['## ' + item['role'].capitalize() + ' — ' + event.get('timestamp', 'timestamp unavailable'), '', '\n\n'.join(messages), ''])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'session_snapshot.jsonl').write_bytes(raw)
    (args.output_dir / 'visible_messages.md').write_text('\n'.join(parts), encoding='utf-8')
    provenance = {'source_path': str(args.source.resolve()), 'session_id': meta['id'],
                  'snapshot_at_utc': timestamp, 'source_bytes': len(raw),
                  'raw_sha256': hashlib.sha256(raw).hexdigest(), 'visible_messages': count,
                  'complete_after_current_turn': False}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
