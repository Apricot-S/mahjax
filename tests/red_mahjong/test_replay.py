from __future__ import annotations

import importlib
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from mahjax.red_mahjong.tenhou import (
    FLAG_DISCARD_TILE_MISMATCH,
    comparable_event_indices,
    iter_rounds,
    mjlog2state,
    normalized_comparison_pairs,
    parse_mjlog,
    replay_round,
    shadow_replay_round,
    snapshot_from_init,
    snapshot_from_state,
)
from mahjax.red_mahjong.env import RedMahjong, _calc_wind
from mahjax.red_mahjong.tile import Tile

replay_scan_mod = importlib.import_module("mahjax.red_mahjong.tenhou.replay_scan")
state_conversion_mod = importlib.import_module("mahjax.red_mahjong.tenhou.state_conversion")


_QUICK_FULL_REPLAY_CASES = [
    ("2021070100gm-00a9-0000-051217a5.mjlog", 0),
    ("2021070100gm-00a9-0000-051217a5.mjlog", 1),
    ("2021070100gm-00a9-0000-051217a5.mjlog", 2),
    ("2021070100gm-00a9-0000-051217a5.mjlog", 3),
    ("2021070100gm-00a9-0000-051217a5.mjlog", 4),
    ("2021070100gm-00a9-0000-051217a5.mjlog", 5),
    ("2021070100gm-00a9-0000-055d72de.mjlog", 0),
    ("2021070100gm-00a9-0000-055d72de.mjlog", 1),
    ("2021070100gm-00a9-0000-055d72de.mjlog", 2),
    ("2021070100gm-00a9-0000-055d72de.mjlog", 3),
    ("2021070100gm-00a9-0000-055d72de.mjlog", 4),
    ("2021070100gm-00a9-0000-055d72de.mjlog", 5),
    ("2021070100gm-00a9-0000-055d72de.mjlog", 6),
    ("2021070100gm-00a9-0000-055d72de.mjlog", 7),
    ("2021070100gm-00a9-0000-0ec27260.mjlog", 0),
    ("2021070100gm-00a9-0000-0ec27260.mjlog", 1),
    ("2021070100gm-00a9-0000-0ec27260.mjlog", 2),
    ("2021070100gm-00a9-0000-0ec27260.mjlog", 3),
    ("2021070100gm-00a9-0000-12c51485.mjlog", 0),
    ("2021070100gm-00a9-0000-12c51485.mjlog", 1),
    ("2021070100gm-00a9-0000-12c51485.mjlog", 2),
    ("2021070100gm-00a9-0000-12c51485.mjlog", 3),
    ("2021070100gm-00a9-0000-12c51485.mjlog", 8),
    ("2021070100gm-00a9-0000-1cd258ec.mjlog", 0),
    ("2021070100gm-00a9-0000-1cd258ec.mjlog", 1),
    ("2021070100gm-00a9-0000-1cd258ec.mjlog", 2),
    ("2021070100gm-00a9-0000-1cd258ec.mjlog", 3),
    ("2021070100gm-00a9-0000-90536b90.mjlog", 3),
    ("2021070100gm-00a9-0000-90536b90.mjlog", 6),
    ("2021070100gm-00a9-0000-a5080734.mjlog", 3),
    ("2021070100gm-00a9-0000-c3e30fad.mjlog", 2),
]
_FILE_SMOKE_REPLAY_CASES = [
    ("2021070100gm-00a9-0000-051217a5.mjlog", 0),
    ("2021070100gm-00a9-0000-055d72de.mjlog", 0),
    ("2021070100gm-00a9-0000-0ec27260.mjlog", 0),
    ("2021070100gm-00a9-0000-12c51485.mjlog", 0),
    ("2021070100gm-00a9-0000-1cd258ec.mjlog", 0),
    ("2021070100gm-00a9-0000-1d273360.mjlog", 0),
    ("2021070100gm-00a9-0000-20acfd6e.mjlog", 0),
    ("2021070100gm-00a9-0000-2d5d509c.mjlog", 0),
    ("2021070100gm-00a9-0000-32da5f03.mjlog", 0),
    ("2021070100gm-00a9-0000-3dd621b6.mjlog", 0),
    ("2021070100gm-00a9-0000-4e664708.mjlog", 0),
    ("2021070100gm-00a9-0000-61debb77.mjlog", 0),
    ("2021070100gm-00a9-0000-7b1e5615.mjlog", 0),
    ("2021070100gm-00a9-0000-7dbeeb8e.mjlog", 0),
    ("2021070100gm-00a9-0000-8ad3d5c9.mjlog", 0),
    ("2021070100gm-00a9-0000-8d2b107a.mjlog", 0),
    ("2021070100gm-00a9-0000-90536b90.mjlog", 0),
    ("2021070100gm-00a9-0000-a5080734.mjlog", 0),
    ("2021070100gm-00a9-0000-a578667b.mjlog", 0),
    ("2021070100gm-00a9-0000-b172fc78.mjlog", 0),
    ("2021070100gm-00a9-0000-b2e49a30.mjlog", 0),
    ("2021070100gm-00a9-0000-b34c84c8.mjlog", 1),
    ("2021070100gm-00a9-0000-b78f8708.mjlog", 0),
    ("2021070100gm-00a9-0000-bb6a19a6.mjlog", 0),
    ("2021070100gm-00a9-0000-bb9b3c32.mjlog", 0),
    ("2021070100gm-00a9-0000-bf152b2b.mjlog", 0),
    ("2021070100gm-00a9-0000-c2bb1ca3.mjlog", 0),
    ("2021070100gm-00a9-0000-c3e30fad.mjlog", 0),
    ("2021070100gm-00a9-0000-d66aa126.mjlog", 0),
    ("2021070100gm-00a9-0000-e0f87bd3.mjlog", 0),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MJLOG_DIR = _REPO_ROOT / "tests" / "red_mahjong" / "assets" / "mjlog"
_MJX_RESOURCE_DIR = _MJLOG_DIR / "cornercases"
def _load_first_round():
    game = parse_mjlog(_MJLOG_DIR / "2021070100gm-00a9-0000-051217a5.mjlog")
    return next(iter_rounds(game))


def _load_round(filename: str, round_index: int):
    game = parse_mjlog(_MJLOG_DIR / filename)
    for idx, round_data in enumerate(iter_rounds(game)):
        if idx == round_index:
            return round_data
    raise AssertionError(f"round {round_index} not found in {filename}")


def _assert_no_flags(result) -> None:
    flagged = [(log.index, log.kind, log.flags) for log in result.logs if log.flags]
    assert flagged == []


def test_init_snapshot_matches_mjlog_init() -> None:
    init_event, events = _load_first_round()

    state = mjlog2state(jax.random.PRNGKey(0), init_event, events, draw_first_tile=False)
    actual = snapshot_from_state(state)
    expected = snapshot_from_init(init_event)

    assert jnp.array_equal(actual.scores, expected.scores)
    assert jnp.array_equal(actual.hand_with_red, expected.hand_with_red)
    assert jnp.array_equal(actual.melds, expected.melds)
    assert jnp.array_equal(actual.meld_counts, expected.meld_counts)
    assert jnp.array_equal(actual.river, expected.river)
    assert jnp.array_equal(actual.discard_counts, expected.discard_counts)
    assert jnp.array_equal(actual.riichi, expected.riichi)
    assert jnp.array_equal(actual.riichi_declared, expected.riichi_declared)
    assert int(actual.dealer) == int(expected.dealer)
    assert int(actual.round) == int(expected.round)
    assert int(actual.honba) == int(expected.honba)
    assert int(actual.kyotaku) == int(expected.kyotaku)
    assert int(actual.current_player) == int(expected.current_player)
    assert jnp.array_equal(actual.dora_indicators, expected.dora_indicators)
    assert int(actual.last_draw) == -1
    assert int(actual.target) == -1
    assert not bool(actual.terminated_round)
    assert not bool(actual.has_won.any())


def test_replay_round_prefix_has_no_critical_mismatch() -> None:
    init_event, events = _load_first_round()
    prefix = events[:16]

    result = replay_round(init_event, prefix, jax.random.PRNGKey(1))

    assert len(result.logs) == len(prefix)
    assert len(result.snapshots) == len(prefix)
    critical = jnp.asarray(
        [log.flags & ~int(FLAG_DISCARD_TILE_MISMATCH) for log in result.logs],
        dtype=jnp.int32,
    )
    assert jnp.all(critical == 0)


def test_replay_snapshots_track_state_progression() -> None:
    init_event, events = _load_first_round()
    prefix = events[:16]

    result = replay_round(init_event, prefix, jax.random.PRNGKey(2))

    first = result.snapshots[0]
    last = result.snapshots[-1]

    assert int(first.hand_with_red[0].sum()) == 14
    assert int(first.current_player) == 0
    assert int(last.discard_counts.sum()) > 0
    assert int(last.hand_with_red.sum()) in (52, 53)


def test_verify_step_matches_step_on_first_discard() -> None:
    init_event, events = _load_first_round()
    state = mjlog2state(jax.random.PRNGKey(4), init_event, events)
    discard_event = next(event for event in events if event.kind == "DISCARD")
    action = int(Tile.from_tile_id_to_tile(jnp.int32(discard_event.attrs["tile"])))

    env = RedMahjong()
    expected = env.step(state, jnp.int32(action))
    actual, is_illegal = env.verify_step(state, jnp.int32(action))

    assert not bool(is_illegal)
    _assert_snapshot_equal(snapshot_from_state(actual), snapshot_from_state(expected))


def _assert_snapshot_equal(lhs, rhs) -> None:
    assert jnp.array_equal(lhs.scores, rhs.scores)
    assert jnp.array_equal(lhs.hand_with_red, rhs.hand_with_red)
    assert jnp.array_equal(lhs.melds, rhs.melds)
    assert jnp.array_equal(lhs.meld_counts, rhs.meld_counts)
    assert jnp.array_equal(lhs.river, rhs.river)
    assert jnp.array_equal(lhs.discard_counts, rhs.discard_counts)
    assert jnp.array_equal(lhs.riichi, rhs.riichi)
    assert jnp.array_equal(lhs.riichi_declared, rhs.riichi_declared)
    assert int(lhs.dealer) == int(rhs.dealer)
    assert int(lhs.round) == int(rhs.round)
    assert int(lhs.honba) == int(rhs.honba)
    assert int(lhs.kyotaku) == int(rhs.kyotaku)
    assert int(lhs.current_player) == int(rhs.current_player)
    assert jnp.array_equal(lhs.dora_indicators, rhs.dora_indicators)
    assert int(lhs.last_draw) == int(rhs.last_draw)
    assert int(lhs.target) == int(rhs.target)
    assert bool(lhs.terminated_round) == bool(rhs.terminated_round)
    assert jnp.array_equal(lhs.has_won, rhs.has_won)


def test_shadow_replay_matches_env_replay_on_comparable_events() -> None:
    init_event, events = _load_first_round()
    prefix = events[:6]
    comparable = comparable_event_indices(prefix)

    actual = replay_round(init_event, prefix, jax.random.PRNGKey(3)).snapshots
    expected = shadow_replay_round(init_event, prefix).snapshots

    assert len(actual) == len(expected)
    assert comparable
    for idx in comparable:
        _assert_snapshot_equal(actual[idx], expected[idx])


def test_normalized_comparison_pairs_follow_verify_step_event_boundaries() -> None:
    init_event, events = _load_first_round()
    prefix = events[:16]
    actual = replay_round(init_event, prefix, jax.random.PRNGKey(6)).snapshots

    pairs = normalized_comparison_pairs(prefix, actual)

    assert pairs == [(0, 0), (1, 2), (3, 4), (5, 5), (6, 6), (7, 8), (9, 10), (11, 12), (13, 14)]


def test_shadow_replay_matches_env_replay_on_normalized_boundaries() -> None:
    init_event, events = _load_first_round()
    prefix = events[:16]

    actual = replay_round(init_event, prefix, jax.random.PRNGKey(5)).snapshots
    expected = shadow_replay_round(init_event, prefix).snapshots
    pairs = normalized_comparison_pairs(prefix, actual)

    assert pairs
    for actual_idx, expected_idx in pairs:
        _assert_snapshot_equal(actual[actual_idx], expected[expected_idx])


def test_high_tile_ids_decode_without_int8_overflow() -> None:
    decoded = [int(Tile.from_tile_id_to_tile(jnp.int32(tile_id))) for tile_id in range(128, 136)]

    assert decoded == [32, 32, 32, 32, 33, 33, 33, 33]


def test_calc_wind_assigns_east_to_dealer() -> None:
    assert _calc_wind(jnp.int32(0)).tolist() == [0, 1, 2, 3]
    assert _calc_wind(jnp.int32(1)).tolist() == [3, 0, 1, 2]
    assert _calc_wind(jnp.int32(2)).tolist() == [2, 3, 0, 1]
    assert _calc_wind(jnp.int32(3)).tolist() == [1, 2, 3, 0]


def test_round_sequence_keeps_rinshan_after_dora_event() -> None:
    init_event, events = _load_round("2021070100gm-00a9-0000-12c51485.mjlog", 2)

    _, rinshan_draws, dora_indicators = state_conversion_mod._build_round_sequences(init_event, events)

    assert rinshan_draws == [5]
    assert dora_indicators == [9, 2]


@pytest.mark.parametrize(
    ("filename", "round_index"),
    _FILE_SMOKE_REPLAY_CASES,
    ids=[f"{filename}:round{round_index}" for filename, round_index in _FILE_SMOKE_REPLAY_CASES],
)
def test_one_clean_round_replay_matches_mjlog_for_many_files(filename: str, round_index: int) -> None:
    init_event, events = _load_round(filename, round_index)

    result = replay_round(init_event, events, jax.random.PRNGKey(round_index))

    _assert_no_flags(result)


@pytest.mark.parametrize(
    ("filename", "round_index"),
    _QUICK_FULL_REPLAY_CASES,
    ids=[f"{filename}:round{round_index}" for filename, round_index in _QUICK_FULL_REPLAY_CASES],
)
def test_full_round_replay_matches_mjlog_quick(filename: str, round_index: int) -> None:
    init_event, events = _load_round(filename, round_index)

    result = replay_round(init_event, events, jax.random.PRNGKey(round_index))

    _assert_no_flags(result)


def test_expected_agari_yaku_parser_ignores_dora_slots() -> None:
    _, events = _load_round("2021070100gm-00a9-0000-051217a5.mjlog", 5)
    agari_event = next(event for event in events if event.kind == "AGARI")

    yaku = replay_scan_mod._expected_agari_yaku(agari_event)

    assert bool(yaku[0])
    assert bool(yaku[1])
    assert not bool(yaku[51])
    assert int(yaku.sum()) == 2


def test_scores_from_sc_returns_final_scores() -> None:
    _, events = _load_round("2021070100gm-00a9-0000-055d72de.mjlog", 1)
    ryuukyoku_event = next(event for event in events if event.kind == "RYUUKYOKU")

    scores = replay_scan_mod._scores_from_sc(ryuukyoku_event.attrs["sc"])

    assert jnp.array_equal(scores, jnp.array([231, 319, 220, 220], dtype=jnp.int32))


@pytest.mark.skipif(
    os.environ.get("RED_MAHJONG_BY_HUMAN_MJLOG_FIRST30_ALL") != "1",
    reason="set RED_MAHJONG_BY_HUMAN_MJLOG_FIRST30_ALL=1 to run full first-30 mjlog replay coverage",
)
def test_first_30_mjlog_files_replay_all_rounds() -> None:
    files = sorted(_MJLOG_DIR.glob("*.mjlog"))[:30]
    assert len(files) == 30
    for path in files:
        for round_index, (init_event, events) in enumerate(iter_rounds(parse_mjlog(path))):
            result = replay_round(init_event, events, jax.random.PRNGKey(round_index))
            assert [log for log in result.logs if log.flags] == [], f"{path.name}:round{round_index}"


@pytest.mark.skipif(
    os.environ.get("RED_MAHJONG_BY_HUMAN_MJLOG_RANGE") != "1",
    reason="set RED_MAHJONG_BY_HUMAN_MJLOG_RANGE=1 with START_INDEX and MAX_FILES to run a slice (e.g. 40-49)",
)
def test_mjlog_replay_range() -> None:
    """Run replay_round for files [START_INDEX : START_INDEX + MAX_FILES]. Use for incremental expansion."""
    all_files = sorted(_MJLOG_DIR.glob("*.mjlog"))
    start_env = os.environ.get("RED_MAHJONG_BY_HUMAN_MJLOG_START_INDEX", "")
    max_env = os.environ.get("RED_MAHJONG_BY_HUMAN_MJLOG_MAX_FILES", "")
    start = int(start_env) if start_env.isdigit() else 0
    max_files = int(max_env) if max_env.isdigit() else 10
    files = all_files[start : start + max_files]
    assert files, f"no files in range start={start} max_files={max_files} (total {len(all_files)} files)"
    for path in files:
        for round_index, (init_event, events) in enumerate(iter_rounds(parse_mjlog(path))):
            result = replay_round(init_event, events, jax.random.PRNGKey(round_index))
            assert [log for log in result.logs if log.flags] == [], f"{path.name}:round{round_index}"


@pytest.mark.skipif(
    os.environ.get("RED_MAHJONG_BY_HUMAN_MJX_RESOURCE_ALL") != "1",
    reason="set RED_MAHJONG_BY_HUMAN_MJX_RESOURCE_ALL=1 to run mjx resource mjlog replay coverage",
)
def test_mjx_resource_mjlog_files_replay_all_rounds() -> None:
    files = sorted(_MJX_RESOURCE_DIR.glob("*.mjlog"))
    assert len(files) == 85
    for path in files:
        for round_index, (init_event, events) in enumerate(iter_rounds(parse_mjlog(path))):
            result = replay_round(init_event, events, jax.random.PRNGKey(round_index))
            assert [log for log in result.logs if log.flags] == [], f"{path.name}:round{round_index}"
