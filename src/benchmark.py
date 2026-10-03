from __future__ import annotations
import json
import tempfile
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config

@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int

def load_conversations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('Dataset must be a list')
    return data

def recall_points(answer: str, expected: list[str]) -> float:
    normalize = lambda text: unicodedata.normalize('NFC', text).casefold()
    return sum(normalize(item) in normalize(answer) for item in expected) / len(expected) if expected else 1.0

def heuristic_quality(answer: str, expected: list[str]) -> float:
    # Proxy for factual coverage; not a claim of independent LLM judging.
    return recall_points(answer, expected) if answer.strip() else 0.0

def run_agent_benchmark(agent_name, agent, conversations, config) -> BenchmarkRow:
    users = {c['user_id'] for c in conversations}
    size = lambda: sum(agent.memory_file_size(u) for u in users) if hasattr(agent, 'memory_file_size') else 0
    before = size()
    threads = set()
    recall, quality = [], []
    for index, conversation in enumerate(conversations):
        user = conversation['user_id']
        thread = f'train-{index}-{conversation["id"]}'
        threads.add(thread)
        for message in conversation['turns']:
            agent.reply(user, thread, message)
        # Evaluate immediately after each conversation, respecting corrections
        # at that point in time. Each question has a fresh independent thread.
        for qi, question in enumerate(conversation.get('recall_questions', [])):
            fresh = f'recall-{index}-{qi}'
            threads.add(fresh)
            answer = agent.reply(user, fresh, question['question'])['response']
            recall.append(recall_points(answer, question['expected_contains']))
            quality.append(heuristic_quality(answer, question['expected_contains']))
    return BenchmarkRow(agent_name, sum(agent.token_usage(t) for t in threads), sum(agent.prompt_token_usage(t) for t in threads), sum(recall) / len(recall) if recall else 0, sum(quality) / len(quality) if quality else 0, size() - before, sum(agent.compaction_count(t) for t in threads))

def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = ['Agent', 'Agent tokens only', 'Prompt tokens processed', 'Cross-session recall', 'Response quality', 'Memory growth (bytes)', 'Compactions']
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        values = [row.agent_name, row.agent_tokens_only, row.prompt_tokens_processed, f'{row.recall_score:.1%}', f'{row.response_quality:.1%}', row.memory_growth_bytes, row.compactions]
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    return '\n'.join(lines)

def main() -> None:
    config = load_config()
    for title, filename in [('Standard Benchmark', 'conversations.json'), ('Long-Context Stress Benchmark', 'advanced_long_context.json')]:
        with tempfile.TemporaryDirectory(prefix='memory-benchmark-', dir=config.state_dir) as directory:
            isolated = replace(config, state_dir=Path(directory), live=False)
            conversations = load_conversations(config.data_dir / filename)
            rows = [run_agent_benchmark(name, cls(isolated, force_offline=True), conversations, isolated) for name, cls in [('Baseline', BaselineAgent), ('Advanced', AdvancedAgent)]]
            print(title)
            print(format_rows(rows))
            print()

if __name__ == '__main__':
    main()
