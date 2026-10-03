from __future__ import annotations
from dataclasses import dataclass
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates, offline_response
from model_provider import build_chat_model
from agent_baseline import SYSTEM_PROMPT

@dataclass
class AgentContext:
    user_id: str
    memory_path: str

class AdvancedAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False):
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / 'profiles')
        self.compact_memory = CompactMemoryManager(self.config.compact_threshold_tokens, self.config.compact_keep_messages)
        self.thread_tokens = {}
        self.thread_prompt_tokens = {}
        self.owners = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str):
        if thread_id in self.owners and self.owners[thread_id] != user_id:
            raise ValueError('Thread belongs to another user')
        self.owners[thread_id] = user_id
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _prompt(self, user_id, thread_id):
        ctx = self.compact_memory.context(thread_id)
        return [{'role': 'system', 'content': SYSTEM_PROMPT + '\nProfile (newest facts take priority):\n' + self.profile_store.read_text(user_id) + '\nThread summary:\n' + ctx['summary']}] + ctx['messages']

    def _reply_offline(self, user_id: str, thread_id: str, message: str):
        for key, value in extract_profile_updates(message, self.config.profile_confidence_threshold).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, 'user', message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        if self.langchain_agent:
            answer = self.langchain_agent.invoke({'messages': self._prompt(user_id, thread_id)})['messages'][-1].text
        else:
            answer = self._offline_response(user_id, thread_id, message)
        tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + tokens
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        self.compact_memory.append(thread_id, 'assistant', answer)
        return {'response': answer, 'agent_tokens': tokens, 'prompt_tokens': prompt_tokens, 'mode': 'live' if self.langchain_agent else 'offline'}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        return sum(estimate_tokens(m['content']) for m in self._prompt(user_id, thread_id))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        ctx = self.compact_memory.context(thread_id)
        facts = {}
        for item in ctx['messages']:
            if item['role'] == 'user':
                facts.update(extract_profile_updates(item['content']))
        facts.update(self.profile_store.facts(user_id))
        return offline_response(message, facts, ctx['summary'])

    def _maybe_build_langchain_agent(self):
        if self.force_offline or not self.config.live:
            return None
        from langchain.agents import create_agent
        # Profile operations and compaction are deterministic middleware executed
        # before invocation, keeping memory policy independent of model/provider.
        return create_agent(build_chat_model(self.config.model))
