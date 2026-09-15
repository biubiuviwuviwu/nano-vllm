from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_sequence(token_ids: list[int]) -> Sequence:
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=4,
        ignore_eos=True,
    )
    return Sequence(token_ids, sampling_params)


def test_new_sequence_starts_waiting():
    seq = make_sequence([10, 20, 30])

    assert seq.status is SequenceStatus.WAITING
    assert seq.num_tokens == 3
    assert seq.num_prompt_tokens == 3
    assert seq.num_completion_tokens == 0
    assert seq.last_token == 30
    assert seq.prompt_token_ids == [10, 20, 30]
    assert seq.completion_token_ids == []



def test_append_token_updates_sequence():
    seq = make_sequence([10,20,30])
    seq.append_token(40)

    assert len(seq) == 4
    assert seq.num_tokens == 4
    assert seq.last_token == 40
    assert seq.num_prompt_tokens == 3
    assert seq.num_completion_tokens == 1
    assert seq.completion_token_ids == [40]


def test_sequence_splits_tokens_into_blocks(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    seq = make_sequence([10, 20, 30, 40, 50])
    assert seq.num_blocks == 2
    assert seq.last_block_num_tokens ==1
    assert seq.block(0) == [10,20,30,40]
    assert seq.block(1) == [50]

