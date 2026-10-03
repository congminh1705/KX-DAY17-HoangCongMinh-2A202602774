from __future__ import annotations
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text.strip()) / 4)

@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        # Hash prevents traversal, Windows reserved names and slug collisions.
        if not user_id:
            raise ValueError('user_id must not be empty')
        return self.root_dir / hashlib.sha256(user_id.encode('utf-8')).hexdigest() / 'User.md'

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding='utf-8') if path.exists() else ''

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(content, encoding='utf-8', newline='\n')
        temporary.replace(path)
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        text = self.read_text(user_id)
        if not search_text or search_text not in text:
            return False
        self.write_text(user_id, text.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return dict(re.findall(r'^- ([a-z_]+): (.+)$', self.read_text(user_id), re.M))

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        if not re.fullmatch(r'[a-z_]+', key) or '\n' in value:
            raise ValueError('Invalid fact')
        facts = self.facts(user_id)
        facts[key] = value
        self.write_text(user_id, '# User profile\n\n' + ''.join(f'- {k}: {v}\n' for k, v in sorted(facts.items())))

@dataclass(frozen=True)
class ProfileUpdate:
    field: str
    value: str
    confidence: float


def extract_profile_candidates(message: str) -> list[ProfileUpdate]:
    updates = {}
    if '?' in message or re.search(r'(nhắc lại|tóm tắt|thử nhớ|con gì)', message, re.I):
        return []
    # Process clauses in order: last explicit assertion wins. Questions,
    # hypotheticals, negations and quoted jokes cannot establish a fact.
    for clause in re.split(r'[.!?;\n]+|,| chứ | nhưng ', message):
        clause = clause.strip()
        lower = clause.lower()
        if not clause or re.search(r'\b(nếu|đùa|có thể|không còn|đừng|chưa|không phải)\b', lower):
            continue
        if re.search(r'(ở đâu|tên gì|nghề gì|là gì|như thế nào|hay là|phải không|còn ở .* không|nhớ .* không)', lower):
            continue
        patterns = {
            'name': r'(?:mình tên(?: là)?|tên mình là|nhắc lại lần cuối cho chắc: tên)\s+([\w]+(?:\s+Stress)?)',
            'location': r'(?:mình\s+(?:(?:hiện tại|hiện|vẫn|đang|giờ|thực ra|từ tuần này)\s+)*(?:đang\s+)?(?:ở|làm việc ở)|hiện ở|nơi ở hiện tại là)\s+([^\s]+(?:\s+[^\s]+)?)',
            'profession': r'(?:đang làm|mình làm|vẫn là|chuyển sang|nghề(?: nghiệp)?(?: hiện tại)?(?: thì)?(?: vẫn)?(?: là)?)\s+(backend engineer|MLOps engineer|product manager)',
            'favorite_drink': r'đồ uống yêu thích(?: của mình)? là\s+(.+)',
            'favorite_food': r'món ăn yêu thích(?: của mình)? là\s+(.+)',
            'pet': r'mình nuôi\s+(.+)',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, clause, re.I)
            if match:
                value = match.group(1).strip()
                if key == 'location':
                    # Do not confuse cafes or travel with residence.
                    place = re.match(r'(Huế|Đà Nẵng|Hà Nội|Hồ Chí Minh|Sài Gòn)(?:\b|$)', value, re.I)
                    if not place:
                        continue
                    value = place.group(1)
                updates[key] = value
        if re.search(r'(mình (?:vẫn )?(?:thích|đang quan tâm)|dài hạn:)', lower):
            interests = [term for term in ('Python', 'AI', 'MLOps', 'RAG') if re.search(r'\b' + term + r'\b', clause, re.I)]
            if interests:
                updates['interests'] = ', '.join(interests)
        if re.search(r'(mình (?:vẫn )?(?:muốn|thích)|hãy trả lời|style trả lời|style.*giữ nguyên|khi giải thích)', lower) and re.search(r'(ngắn|bullet|gọn)', lower):
            updates['response_style'] = 'ngắn gọn' + (' theo 3 bullet' if '3 bullet' in lower else ' thành bullet' if 'bullet' in lower else '')
    # Style qualifiers may be in following comma-separated clauses.
    if 'interests' in updates:
        updates['interests'] = ', '.join(term for term in ('Python', 'AI', 'MLOps', 'RAG') if re.search(r'\b' + term + r'\b', message, re.I))
    if 'response_style' in updates:
        if 'ví dụ' in message.lower():
            updates['response_style'] += ', có ví dụ thực tế/thực chiến'
        if 'trade-off' in message.lower():
            updates['response_style'] += ', nhấn trade-off'
    return [ProfileUpdate(k, v, 0.95) for k, v in updates.items()]


def extract_profile_updates(message: str, confidence_threshold: float = 0.85) -> dict[str, str]:
    return {u.field: u.value for u in extract_profile_candidates(message) if u.confidence >= confidence_threshold}


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    facts = {}
    snippets = []
    for msg in messages:
        if msg['role'] == 'user':
            facts.update(extract_profile_updates(msg['content']))
            snippets.append(msg['content'][:180])
        elif msg['role'] == 'system':
            snippets.append(msg['content'])
    # Keep structured facts and bounded snippets, never concatenate full history.
    return ('; '.join(f'{k}: {v}' for k, v in facts.items()) + '\n' + '\n'.join(snippets[-max_items:]))[:1400]

@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self):
        if self.threshold_tokens <= 0 or self.keep_messages < 1:
            raise ValueError('Invalid compact settings')

    def append(self, thread_id: str, role: str, content: str) -> None:
        state = self.state.setdefault(thread_id, {'messages': [], 'summary': '', 'compactions': 0})
        messages = state['messages']
        messages.append({'role': role, 'content': content})
        tokens = estimate_tokens(state['summary']) + sum(estimate_tokens(m['content']) for m in messages)
        if tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            older = messages[:-self.keep_messages]
            prior = [{'role': 'system', 'content': state['summary']}] if state['summary'] else []
            summary = summarize_messages(prior + older)
            # A bounded summary ensures repeated compaction does not grow forever.
            state['summary'] = summary[:max(64, min(1400, self.threshold_tokens * 2))]
            state['messages'] = messages[-self.keep_messages:]
            state['compactions'] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        state = self.state.get(thread_id, {'messages': [], 'summary': '', 'compactions': 0})
        return {**state, 'messages': [dict(m) for m in state['messages']]}

    def compaction_count(self, thread_id: str) -> int:
        return self.context(thread_id)['compactions']


def offline_response(message: str, facts: dict[str, str], context: str = '') -> str:
    lower = message.lower()
    if '?' in message or re.search(r'(nhắc lại|tóm tắt|thử nhớ|tên gì|ở đâu)', lower):
        selectors = {'name': ('tên', 'ai', 'tóm tắt'), 'location': ('ở đâu', 'nơi ở', 'huế', 'hà nội'), 'profession': ('nghề', 'tóm tắt'), 'favorite_drink': ('đồ uống',), 'favorite_food': ('món ăn',), 'pet': ('nuôi', 'con gì'), 'response_style': ('style', 'kiểu trả lời', 'như thế nào'), 'interests': ('quan tâm', 'tóm tắt')}
        chosen = [f'{key}: {facts[key]}' for key, terms in selectors.items() if key in facts and any(term in lower for term in terms)]
        if chosen:
            if '3 bullet' in facts.get('response_style', ''):
                groups = [chosen[i::3] for i in range(3)]
                return '\n'.join('- ' + '; '.join(g or ['Thông tin chưa được cung cấp.']) for g in groups)
            return '; '.join(chosen) + '.'
        if context and 'chủ đề' in lower:
            return context[:300]
        return 'Mình chưa có thông tin đó trong ngữ cảnh hiện tại.'
    return 'Đã nhận thông tin. Mình sẽ dùng ngữ cảnh này cho các câu hỏi tiếp theo.'
