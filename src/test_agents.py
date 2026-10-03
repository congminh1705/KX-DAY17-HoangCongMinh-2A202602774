from dataclasses import replace
from pathlib import Path
import pytest
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import load_conversations, run_agent_benchmark
from config import load_config
from memory_store import UserProfileStore, CompactMemoryManager, extract_profile_updates, estimate_tokens
from model_provider import normalize_provider

def make_config(tmp_path: Path):
    return replace(load_config(), state_dir=tmp_path, compact_threshold_tokens=500, compact_keep_messages=2, live=False)

def test_user_markdown_read_write_edit(tmp_path):
    store = UserProfileStore(tmp_path)
    assert store.read_text('u') == ''
    path = store.write_text('u', '# User\nTên: Minh\n')
    assert path.name == 'User.md'
    assert store.edit_text('u', 'Minh', 'Dũng')
    assert 'Dũng' in store.read_text('u')
    assert not store.edit_text('u', 'missing', 'x')
    assert store.file_size('u') == len(store.read_text('u').encode('utf-8'))
    assert store.path_for('../u').is_relative_to(tmp_path)
    assert store.path_for('a/b') != store.path_for('a_b')

def test_compact_trigger(tmp_path):
    memory = CompactMemoryManager(200, 2)
    for i in range(12):
        memory.append('t', 'user', f'Lượt {i}: ' + 'nội dung ' * 100)
    ctx = memory.context('t')
    assert memory.compaction_count('t') > 1
    assert len(ctx['messages']) == 2
    assert ctx['summary']
    assert len(ctx['summary']) <= 400
    ctx['messages'].clear()
    assert memory.context('t')['messages']

def test_cross_session_recall(tmp_path):
    config = make_config(tmp_path)
    for cls in (BaselineAgent, AdvancedAgent):
        agent = cls(config, True)
        agent.reply('u', 'one', 'Mình tên là DũngCT. Mình ở Huế và đang làm MLOps engineer.')
        assert 'DũngCT' in agent.reply('u', 'one', 'Mình tên gì?')['response']
        answer = agent.reply('u', 'two', 'Mình tên gì?')['response']
        assert ('DũngCT' in answer) == (cls is AdvancedAgent)
    recreated = AdvancedAgent(config, True)
    assert 'DũngCT' in recreated.reply('u', 'new', 'Mình tên gì?')['response']
    assert 'DũngCT' not in recreated.reply('other', 'other', 'Mình tên gì?')['response']
    with pytest.raises(ValueError):
        recreated.reply('other', 'new', 'Xin chào')

def test_compact_reduces_prompt_load_on_long_thread(tmp_path):
    config = make_config(tmp_path)
    agents = [BaselineAgent(config, True), AdvancedAgent(config, True)]
    for agent in agents:
        for i in range(30):
            agent.reply('u', 'long', f'Lượt {i}. ' + 'Đây là ngữ cảnh tạm thời rất dài. ' * 80)
    assert agents[1].compaction_count('long') > 1
    assert agents[1].prompt_token_usage('long') < agents[0].prompt_token_usage('long') * 0.5

@pytest.mark.parametrize('text', ['Mình tên gì?', 'Nhắc lại tên và nghề của mình.', 'Nếu mình ở Hà Nội thì sao?', 'Mình không còn làm backend engineer nữa.', 'Mình đùa rằng mình làm product manager.', 'Hà Nội chỉ là nơi mình đi họp.', 'Mình nuôi con gì?'])
def test_no_false_facts(text):
    assert extract_profile_updates(text) == {}

def test_confidence_and_corrections(tmp_path):
    assert extract_profile_updates('Mình tên là Minh.', 0.99) == {}
    agent = AdvancedAgent(make_config(tmp_path), True)
    agent.reply('u', 'a', 'Mình ở Đà Nẵng và đang làm backend engineer.')
    agent.reply('u', 'a', 'Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.')
    agent.reply('u', 'a', 'Giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.')
    facts = agent.profile_store.facts('u')
    assert facts['location'] == 'Huế'
    assert facts['profession'] == 'MLOps engineer'
    assert 'backend engineer' not in agent.profile_store.read_text('u')
    size = agent.memory_file_size('u')
    for _ in range(10):
        agent.reply('u', 'a', 'Mình đang ở Huế.')
    assert agent.memory_file_size('u') == size

@pytest.mark.parametrize('filename', ['conversations.json', 'advanced_long_context.json'])
def test_full_datasets(tmp_path, filename):
    config = make_config(tmp_path)
    data = load_conversations(config.data_dir / filename)
    baseline = run_agent_benchmark('Baseline', BaselineAgent(config, True), data, config)
    advanced = run_agent_benchmark('Advanced', AdvancedAgent(config, True), data, config)
    assert baseline.recall_score == 0
    assert advanced.recall_score == 1
    assert advanced.memory_growth_bytes > 0
    if 'long_context' in filename:
        assert advanced.compactions > 1
        assert advanced.prompt_tokens_processed < baseline.prompt_tokens_processed

@pytest.mark.parametrize('provider', ['openai', 'custom', 'gemini', 'anthropic', 'ollama', 'openrouter'])
def test_provider_dispatch_without_network(provider, monkeypatch):
    import model_provider
    from types import SimpleNamespace
    def fake_import(name):
        return SimpleNamespace(**{cls: lambda **kw: kw for cls in ['ChatOpenAI', 'ChatGoogleGenerativeAI', 'ChatAnthropic', 'ChatOllama', 'ChatOpenRouter']})
    monkeypatch.setattr(model_provider, 'import_module', fake_import)
    result = model_provider.build_chat_model(model_provider.ProviderConfig(provider, 'test', api_key='fake', base_url='http://localhost:1234'))
    assert result['model'] == 'test'
    assert normalize_provider('anthorpic') == 'anthropic'

def test_token_estimator():
    assert estimate_tokens('  ') == 0
    assert estimate_tokens('12345') == 2
