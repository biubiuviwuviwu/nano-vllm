from types import SimpleNamespace

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.scheduler import Scheduler


def make_sequence(token_ids: list[int]) -> Sequence:
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=4,
        ignore_eos=True,
    )
    return Sequence(token_ids, sampling_params)


def make_scheduler(
    monkeypatch,
    max_num_batched_tokens: int = 32,
    max_num_seqs: int = 4,
    num_kvcache_blocks: int = 2,
    kvcache_block_size: int = 4,
) -> Scheduler:
    monkeypatch.setattr(Sequence, "block_size", kvcache_block_size)

    config = SimpleNamespace(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        num_kvcache_blocks=num_kvcache_blocks,
        kvcache_block_size=kvcache_block_size,
        eos=2,
    )
    return Scheduler(config)


def test_add_request(monkeypatch):
    token_ids = [10, 20, 30, 40, 50, 60]
    seq = make_sequence(token_ids)
    scheduler = make_scheduler(monkeypatch)

    scheduler.add(seq)

    assert len(scheduler.waiting) == 1
    assert scheduler.waiting[0] is seq
    assert len(scheduler.running) == 0
    assert seq.status is SequenceStatus.WAITING


def test_chunked_prefill(monkeypatch):
    seq = make_sequence([10, 20, 30, 40, 50, 60])
    scheduler = make_scheduler(monkeypatch, max_num_batched_tokens=4)
    scheduler.add(seq)

    # 第一轮调度 4 个 Prompt Token。
    scheduled, is_prefill = scheduler.schedule()

    assert len(scheduled) == 1
    assert scheduled[0] is seq
    assert seq.num_scheduled_tokens == 4
    assert seq.num_cached_tokens == 0
    assert seq.status is SequenceStatus.WAITING
    assert is_prefill is True

    scheduler.postprocess(scheduled, [101], is_prefill)

    assert seq.num_cached_tokens == 4

    # 第二轮调度剩余的 2 个 Prompt Token。
    scheduled, is_prefill = scheduler.schedule()

    assert len(scheduled) == 1
    assert scheduled[0] is seq
    assert seq.status is SequenceStatus.RUNNING
    assert seq.num_scheduled_tokens == 2
    assert is_prefill is True

    scheduler.postprocess(scheduled, [102], is_prefill)

    assert seq.num_cached_tokens == 6
    assert len(scheduler.waiting) == 0
    assert list(scheduler.running) == [seq]


def test_decode_one_token(monkeypatch):
    seq = make_sequence([10, 20, 30, 40, 50, 60])
    scheduler = make_scheduler(monkeypatch, max_num_batched_tokens=4)
    scheduler.add(seq)

    # 完成两轮 Prefill。
    scheduled, is_prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [101], is_prefill)

    scheduled, is_prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [102], is_prefill)

    # Decode 每轮只调度一个 Token。
    scheduled, is_prefill = scheduler.schedule()

    assert len(scheduled) == 1
    assert scheduled[0] is seq
    assert seq.num_scheduled_tokens == 1
    assert seq.status is SequenceStatus.RUNNING
    assert is_prefill is False


def test_kvcache_full_preempts_tail_sequence(monkeypatch):
    scheduler = make_scheduler(
        monkeypatch,
        max_num_batched_tokens=8,
        max_num_seqs=2,
        num_kvcache_blocks=2,
        kvcache_block_size=4,
    )

    seq_a = make_sequence([10, 20, 30, 40])
    seq_b = make_sequence([20, 40, 60, 80])

    scheduler.add(seq_a)
    scheduler.add(seq_b)

    # 两个请求各占用一个 Block，并完成 Prefill。
    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is True
    assert scheduled == [seq_a, seq_b]

    scheduler.postprocess(scheduled, [101, 102], is_prefill)

    assert list(scheduler.running) == [seq_a, seq_b]
    assert len(scheduler.waiting) == 0
    assert len(scheduler.block_manager.free_block_ids) == 0

    # 两个 Sequence 的长度都是 5。
    # 在 block_size=4 时，下一轮 Decode 需要申请新 Block。
    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is False
    assert scheduled == [seq_a]

    # seq_b 是队尾请求，因此被抢占。
    assert seq_a.status is SequenceStatus.RUNNING
    assert seq_b.status is SequenceStatus.WAITING

    assert list(scheduler.running) == [seq_a]
    assert list(scheduler.waiting) == [seq_b]

    # seq_b 的 Block 已被释放，并被 seq_a 用于后续 Decode。
    assert seq_b.block_table == []
    assert seq_b.num_cached_tokens == 0
    assert len(seq_a.block_table) == 2
    assert scheduler.block_manager.used_block_ids == set(seq_a.block_table)
