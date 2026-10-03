from __future__ import annotations
from dataclasses import dataclass, field
from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates, offline_response
from model_provider import build_chat_model

SYSTEM_PROMPT = 'Bạn là trợ lý tiếng Việt. Chỉ dùng facts được cung cấp, nói rõ khi chưa biết.'

@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0

class BaselineAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False):
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions = {}
        self.owners = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str):
        if thread_id in self.owners and self.owners[thread_id] != user_id:
            raise ValueError('Thread belongs to another user')
        self.owners[thread_id] = user_id
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str):
        state = self.sessions.setdefault(thread_id, SessionState())
        state.messages.append({'role': 'user', 'content': message})
        prompt = [{'role': 'system', 'content': SYSTEM_PROMPT}] + state.messages
        prompt_tokens = sum(estimate_tokens(m['content']) for m in prompt)
        if self.langchain_agent:
            answer = self.langchain_agent.invoke({'messages': prompt})['messages'][-1].text
        else:
            facts = {}
            for item in state.messages:
                if item['role'] == 'user':
                    facts.update(extract_profile_updates(item['content']))
            answer = offline_response(message, facts)
        tokens = estimate_tokens(answer)
        state.token_usage += tokens
        state.prompt_tokens_processed += prompt_tokens
        state.messages.append({'role': 'assistant', 'content': answer})
        return {'response': answer, 'agent_tokens': tokens, 'prompt_tokens': prompt_tokens, 'mode': 'live' if self.langchain_agent else 'offline'}

    def _maybe_build_langchain_agent(self):
        if self.force_offline or not self.config.live:
            return None
        from langchain.agents import create_agent
        # Explicit history is the sole short-term state; no duplicate checkpointer.
        return create_agent(build_chat_model(self.config.model))
