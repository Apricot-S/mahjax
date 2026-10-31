# Copyright 2025 The Mahjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for showing states handed in from Python (``mahjax.ui.show``)."""

from __future__ import annotations

import base64
import functools
import gzip
import html
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mahjax.mahjong_3p.state import State as ThreePlayerState
from mahjax.ui import save_html, show, standalone
from mahjax.ui.agents import _rule_based
from mahjax.ui.record import StateReplay
from mahjax.ui.rules import rules_for, shared_env, shared_step_fn


@functools.lru_cache(maxsize=None)
def _rollout() -> List[Any]:
    """The first round of a rule-based east game and the opening of the second, every state kept.

    The env and its traced step are the ones the other UI tests use.
    """
    env, step, act = shared_env("red_4p", "east"), shared_step_fn("red_4p", "east"), _rule_based()
    key = jax.random.PRNGKey(0)
    state = jax.device_get(env.init(key))
    first_deal = (int(state.round_state.round), int(state.round_state.honba))
    states = [state]
    for _ in range(400):
        key, k1, k2 = jax.random.split(key, 3)
        state = jax.device_get(step(state, act(state, k1), k2))
        states.append(state)
        deal = (int(state.round_state.round), int(state.round_state.honba))
        if deal != first_deal and int(state.round_state.round_step) > 0:
            break
    return states


def _as_auto(states: List[Any]) -> List[Any]:
    """The states an ``round_end="auto"`` env shows: it deals the next round in the
    step that ends one, so the round's end and the sharing steps are never seen."""
    rules = rules_for("red_4p")
    return [state for state in states if not rules.is_round_over(state)]


def test_a_confirm_game_names_each_action_and_shows_the_result() -> None:
    # The game is made to end in the second round, so only that round's result ends it.
    states = _rollout()
    replay = StateReplay(states + [states[-1].replace(terminated=np.bool_(True))])
    assert replay.rules.env_id == "red_4p"
    assert sum(1 for step in replay._steps if step["event"]) > 0
    first, last = sorted(replay._results)
    assert replay.frame(first, viewpoint=None, show_all=True)["result"] is not None
    assert [replay._results[i]["gameOver"] for i in (first, last)] == [False, True]
    # Two rounds tile the frames; the sharing steps after a result are not frames.
    assert [span.start for span in replay.rounds] == [0, first + 1]
    assert replay.rounds[0].end == first and replay.rounds[-1].end == replay.total - 1 == last


def test_an_auto_game_plays_back_every_state() -> None:
    states = _as_auto(_rollout())
    replay = StateReplay(states)
    assert replay.total == len(states)
    assert len(replay.rounds) == 2 and not replay._results
    assert len(replay.frames(0, replay.total)) == replay.total


def test_a_stacked_state_is_a_sequence() -> None:
    states = _rollout()[:10]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
    assert StateReplay(stacked).total == StateReplay(states).total
    assert StateReplay(states[0]).total == 1


def test_a_batch_of_sequences_is_refused() -> None:
    states = _rollout()[:4]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
    trajectory = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), batched, batched)
    with pytest.raises(ValueError, match="unbatched"):
        StateReplay(trajectory)


def test_three_player_states_are_not_shown_yet() -> None:
    with pytest.raises(NotImplementedError):
        StateReplay(ThreePlayerState())


def _carried(page: str) -> Dict[str, Any]:
    """The data a saved page carries, unpacked as its script unpacks it."""
    (packed,) = re.findall(r'<script id="mahjax-data" type="application/octet-stream">([^<]*)</script>', page)
    return json.loads(gzip.decompress(base64.b64decode(packed)))


def test_outside_a_notebook_show_saves_a_page_that_carries_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    names = ["A", "B", "C", "D"]
    show(_rollout(), names=names)
    (path,) = tmp_path.glob("mahjax-*.html")
    assert str(path) in capsys.readouterr().out
    page = path.read_text(encoding="utf-8")
    assert "/static/" not in page  # the script and the styles are inline
    data = _carried(page)
    replay = StateReplay(_rollout(), names=names)
    assert data["frames"] == json.loads(json.dumps(replay.frames(0, replay.total)))
    assert [seat["name"] for seat in data["replay"]["seats"]] == names
    assert {name: len(files) for name, files in data["tiles"].items()} == {"ja": 40, "en": 40}


def test_save_html_writes_the_page(tmp_path: Path) -> None:
    save_html(_rollout()[:3], tmp_path / "game.html")
    assert _carried((tmp_path / "game.html").read_text(encoding="utf-8"))["replay"]["total"] == 3


def test_a_notebook_gets_the_page_in_an_iframe() -> None:
    page = "<!DOCTYPE html><p class=\"x\">a & 'b' < c</p>"
    found = re.fullmatch(r'<iframe srcdoc="([^"]*)" style="([^"]*)"></iframe>', standalone._iframe(page, 500))
    assert html.unescape(found.group(1)) == page
    assert "height: 500px" in found.group(2)
