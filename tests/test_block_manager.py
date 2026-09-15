from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def make_sequence(token_ids: list[int]) -> Sequence:
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=4,
        ignore_eos=True,
    )
    return Sequence(token_ids, sampling_params)

def test_allocate_and_deallocate_updates_block_ownership(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)

    manager = BlockManager(num_blocks=4, block_size=4)

    seq = make_sequence([10, 20, 30, 40, 50])

    num_cached_blocks = manager.can_allocate(seq)

    assert num_cached_blocks == 0

    manager.allocate(seq, num_cached_blocks)

    assert len(seq.block_table) == 2

    assert manager.used_block_ids == set(seq.block_table)

    assert len(manager.free_block_ids) == 2

    for block_id in seq.block_table:
        assert manager.blocks[block_id].ref_count == 1

    manager.deallocate(seq)

    assert len(seq.block_table) == 0
    assert len(manager.used_block_ids) == 0
    assert len(manager.free_block_ids) == 4

    for blk in manager.blocks:
        assert blk.ref_count == 0


def test_sequences_share_a_cached_prefix_block(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    manager = BlockManager(num_blocks=6, block_size=4)

    seq_a = make_sequence([10,20,30,40,50])

    num_cached_blocks = manager.can_allocate(seq_a)

    assert num_cached_blocks == 0

    manager.allocate(seq_a, num_cached_blocks)

    cached_block_id = seq_a.block_table[0]

    seq_a.num_scheduled_tokens = seq_a.num_tokens

    manager.hash_blocks(seq_a)

    # manager.deallocate(seq_a)



    seq_b = make_sequence([10, 20, 30, 40, 99])

    num_cached_blocks = manager.can_allocate(seq_b)

    assert num_cached_blocks == 1

    manager.allocate(seq_b, num_cached_blocks)

    assert seq_b.block_table[0] == cached_block_id

    assert manager.blocks[cached_block_id].ref_count == 2

    manager.deallocate(seq_a)

    assert manager.blocks[cached_block_id].ref_count == 1

    manager.deallocate(seq_b)

    assert manager.blocks[cached_block_id].ref_count == 0
    assert cached_block_id in manager.free_block_ids
    assert len(manager.used_block_ids) == 0
