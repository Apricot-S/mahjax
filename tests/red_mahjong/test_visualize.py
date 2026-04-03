from pathlib import Path

from mahjax.red_mahjong.state import default_state
from mahjax.red_mahjong.visualization import (
    render_round_svg,
    render_svg_animation,
    save_play_history_svg,
)


def _collect_states(count: int = 10) -> list:
    state = default_state()
    return [state for _ in range(max(1, count))]


def test_render_round_svg_contains_svg_tag() -> None:
    state = _collect_states(0)[0]
    svg = render_round_svg(state, show_all_hands=True)
    assert svg.startswith("<svg")
    assert "</svg>" in svg


def test_render_round_svg_supports_english_language() -> None:
    state = _collect_states(0)[0]
    svg_ja = render_round_svg(state, show_all_hands=True, language="ja")
    svg_en = render_round_svg(state, show_all_hands=True, language="en")
    assert "東1局" in svg_ja
    assert "East 1" in svg_en
    assert svg_ja != svg_en


def test_save_play_history_svg_for_10_steps() -> None:
    states = _collect_states(10)
    out = Path("fig/red_mahjong_10steps_test.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_play_history_svg(states, out, columns=5, show_all_hands=True)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_svg_animation_contains_keyframes() -> None:
    states = _collect_states(5)
    svg = render_svg_animation(states, frame_duration_seconds=0.1, show_all_hands=True)
    assert "@keyframes" in svg
