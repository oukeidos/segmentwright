from segmentwright.chunking import ChunkPlannerConfig, plan_chunks, silence_gap_candidates
from segmentwright.vad.base import VadSegment


def test_silence_gap_candidates_skip_edges() -> None:
    speech = [
        VadSegment(10, 20),
        VadSegment(30, 40),
    ]

    candidates = silence_gap_candidates(
        duration_sec=50,
        speech_segments=speech,
        min_silence_sec=0.1,
    )

    assert len(candidates) == 1
    assert candidates[0].time_sec == 25
    assert candidates[0].source == "vad_silence_gap"


def test_plan_chunks_respects_hard_max_with_forced_anchor() -> None:
    plan = plan_chunks(
        duration_sec=650,
        speech_segments=[VadSegment(0, 650)],
        source_id="source",
        config=ChunkPlannerConfig(target_chunk_sec=300, max_chunk_sec=300),
    )

    assert plan.split_times == [300.0, 600.0]
    assert all(chunk.duration_sec <= 300 for chunk in plan.chunks)
    assert [chunk.start_sample for chunk in plan.chunks] == [0, 4_800_000, 9_600_000]


def test_plan_chunks_uses_silence_candidate() -> None:
    plan = plan_chunks(
        duration_sec=500,
        speech_segments=[
            VadSegment(0, 240),
            VadSegment(260, 500),
        ],
        source_id="source",
        config=ChunkPlannerConfig(target_chunk_sec=250, max_chunk_sec=300),
    )

    assert plan.split_times == [250.0]
    assert len(plan.chunks) == 2
    assert plan.chunks[1].cut_source == "vad_silence_gap"
