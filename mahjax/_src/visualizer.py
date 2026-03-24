# NOTE: This file is copied and modified from Pgx (https://github.com/sotetsuk/pgx).
# Copyright belongs to the original authors.
# We keep tracking the updates of original Pgx implementation.
# We try to minimize the modification to this file. Exceptions includes:
#   - remove unnesesary env implementation since mahjax only support mahjong environment.
#
# Copyright 2023 The Pgx Authors.
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

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import jax
import svgwrite  # type: ignore

from mahjax.core import State

ColorTheme = Literal["light", "dark"]


@dataclass
class Config:
    color_theme: ColorTheme = "light"
    scale: float = 1.0
    frame_duration_seconds: float = 0.2


global_config = Config()


def set_visualization_config(
    *,
    color_theme: ColorTheme = "light",
    scale: float = 1.0,
    frame_duration_seconds: float = 0.2,
):
    global_config.color_theme = color_theme
    global_config.scale = scale
    global_config.frame_duration_seconds = frame_duration_seconds


@dataclass
class ColorSet:
    p1_color: str = "black"
    p2_color: str = "white"
    p1_outline: str = "black"
    p2_outline: str = "black"
    background_color: str = "white"
    grid_color: str = "black"
    text_color: str = "black"


class Visualizer:
    """The Visualizer class (copied and modified from Pgx)

    color_theme: Default(None) is "light"
    scale: change image size. Default(None) is 1.0
    """

    def __init__(
        self,
        *,
        color_theme: Optional[ColorTheme] = None,
        scale: Optional[float] = None,
    ) -> None:
        color_theme = (
            color_theme if color_theme is not None else global_config.color_theme
        )
        scale = scale if scale is not None else global_config.scale

        self.config = {
            "GRID_SIZE": -1,
            "BOARD_WIDTH": -1,
            "BOARD_HEIGHT": -1,
            "COLOR_THEME": color_theme,
            "COLOR_SET": ColorSet(),
            "SCALE": scale,
        }
        self._make_dwg_group = None

    """
    notebook で可視化する際に、変数名のみで表示させる場合
    def _repr_html_(self) -> str:
        assert self.state is not None
        return self._to_dwg_from_states(states=self.state).tostring()
    """

    def get_dwg(
        self,
        states,
        use_english=False,
    ):
        try:
            SIZE = len(states.current_player)
            WIDTH = math.ceil(math.sqrt(SIZE - 0.1))
            if SIZE - (WIDTH - 1) ** 2 >= WIDTH:
                HEIGHT = WIDTH
            else:
                HEIGHT = WIDTH - 1
            if SIZE == 1:
                states = self._get_nth_state(states, 0)
        except TypeError:
            SIZE = 1
            WIDTH = 1
            HEIGHT = 1

        self._set_config_by_state(states, use_english=use_english)
        assert self._make_dwg_group is not None

        GRID_SIZE = self.config["GRID_SIZE"]
        BOARD_WIDTH = self.config["BOARD_WIDTH"]
        BOARD_HEIGHT = self.config["BOARD_HEIGHT"]
        SCALE = self.config["SCALE"]

        canvas_width = (BOARD_WIDTH + 1) * GRID_SIZE * WIDTH * SCALE
        canvas_height = (BOARD_HEIGHT + 1) * GRID_SIZE * HEIGHT * SCALE

        dwg = svgwrite.Drawing(
            "temp.svg",
            (
                canvas_width,
                canvas_height,
            ),
        )
        dwg.attribs["viewBox"] = f"0 0 {canvas_width} {canvas_height}"
        group = dwg.g()

        # background
        group.add(
            dwg.rect(
                (0, 0),
                (
                    (BOARD_WIDTH + 1) * GRID_SIZE * WIDTH,
                    (BOARD_HEIGHT + 1) * GRID_SIZE * HEIGHT,
                ),
                fill=self.config["COLOR_SET"].background_color,
            )
        )

        if SIZE == 1:
            g = self._make_dwg_group(dwg, states, self.config)
            g.translate(
                GRID_SIZE * 1 / 2,
                GRID_SIZE * 1 / 2,
            )
            group.add(g)
            group.scale(SCALE)
            dwg.add(group)
            return dwg

        for i in range(SIZE):
            x = i % WIDTH
            y = i // WIDTH
            _state = self._get_nth_state(states, i)
            g = self._make_dwg_group(
                dwg,
                _state,  # type:ignore
                self.config,
            )

            g.translate(
                GRID_SIZE * 1 / 2 + (BOARD_WIDTH + 1) * GRID_SIZE * x,
                GRID_SIZE * 1 / 2 + (BOARD_HEIGHT + 1) * GRID_SIZE * y,
            )
            group.add(g)
            group.add(
                dwg.rect(
                    (
                        (BOARD_WIDTH + 1) * GRID_SIZE * x,
                        (BOARD_HEIGHT + 1) * GRID_SIZE * y,
                    ),
                    (
                        (BOARD_WIDTH + 1) * GRID_SIZE,
                        (BOARD_HEIGHT + 1) * GRID_SIZE,
                    ),
                    fill="none",
                    stroke="gray",
                )
            )
        group.scale(SCALE)
        dwg.add(group)
        return dwg

    def _set_config_by_state(self, _state: State, use_english=False):  # noqa: C901
        del use_english
        raise NotImplementedError(
            "DWG-based board renderer is removed for mahjong. Use SVG-based renderer via state.to_svg()/save_svg()."
        )

    def _get_nth_state(self, states: State, i):
        return jax.tree_util.tree_map(lambda x: x[i], states)


def _to_red_env_state(state: State):
    """Convert no_red style state to red visualizer state."""
    from mahjax.red_mahjong.state import default_state

    if hasattr(state, "players") and hasattr(state, "round_state"):
        return state

    rs = default_state()
    hand34 = state._hand.astype("int8")
    hand37 = jax.numpy.zeros((4, 37), dtype=jax.numpy.int8).at[:, :34].set(hand34)
    legal_4p = state._legal_action_mask_4p
    if legal_4p.shape[1] < rs.players.legal_action_mask.shape[1]:
        pad = rs.players.legal_action_mask.shape[1] - legal_4p.shape[1]
        legal_4p = jax.numpy.pad(legal_4p, ((0, 0), (0, pad)), constant_values=False)
    legal_1p = state.legal_action_mask
    if legal_1p.shape[0] < rs.legal_action_mask.shape[0]:
        pad = rs.legal_action_mask.shape[0] - legal_1p.shape[0]
        legal_1p = jax.numpy.pad(legal_1p, (0, pad), constant_values=False)
    return rs.replace(
        current_player=state.current_player,
        legal_action_mask=legal_1p,
        players=rs.players.replace(
            hand=hand34,
            hand_with_red=hand37,
            legal_action_mask=legal_4p,
            melds=state._melds,
            meld_counts=state._n_meld,
            river=state._river,
            discard_counts=state._n_river,
            riichi=state._riichi,
        ),
        round_state=rs.round_state.replace(
            round=state._round,
            honba=state._honba,
            dealer=state._dealer,
            score=state._score.astype(jax.numpy.int32),
            next_deck_ix=state._next_deck_ix.astype(jax.numpy.int32),
            last_deck_ix=state._last_deck_ix.astype(jax.numpy.int8),
            dora_indicators=state._dora_indicators,
            last_draw=state._last_draw,
            target=state._target,
        ),
        rewards=state.rewards,
        terminated=state.terminated,
        truncated=state.truncated,
    )


def save_svg(
    state: State,
    filename: Union[str, Path],
    *,
    color_theme: Optional[Literal["light", "dark"]] = None,
    scale: Optional[float] = None,
    use_english: bool = False,
) -> None:
    if state.env_id.startswith("minatar"):
        state.save_svg(filename=filename)
    elif state.env_id == "mahjong":
        from mahjax.red_mahjong.visualization import save_svg as save_svg_assets

        save_svg_assets(_to_red_env_state(state), filename)
    else:
        v = Visualizer(color_theme=color_theme, scale=scale)
        v.get_dwg(states=state, use_english=use_english).saveas(filename)


def save_svg_animation(
    states: Sequence[State],
    filename: Union[str, Path],
    *,
    color_theme: Optional[Literal["light", "dark"]] = None,
    scale: Optional[float] = None,
    frame_duration_seconds: Optional[float] = None,
    use_english: bool = False,
) -> None:
    assert not states[0].env_id.startswith(
        "minatar"
    ), "MinAtar does not support svg animation."
    if states[0].env_id == "mahjong":
        from mahjax.red_mahjong.visualization import save_svg_animation as save_svg_animation_assets

        save_svg_animation_assets([_to_red_env_state(s) for s in states], filename, frame_duration_seconds=frame_duration_seconds or global_config.frame_duration_seconds)
        return
    v = Visualizer(color_theme=color_theme, scale=scale)

    if frame_duration_seconds is None:
        frame_duration_seconds = global_config.frame_duration_seconds

    frame_groups = []
    dwg = None
    for i, state in enumerate(states):
        dwg = v.get_dwg(states=state, use_english=use_english)
        assert (
            len([e for e in dwg.elements if type(e) is svgwrite.container.Group]) == 1
        ), "Drawing must contain only one group"
        group: svgwrite.container.Group = dwg.elements[-1]
        group["id"] = f"_fr{i:x}"  # hex frame number
        group["class"] = "frame"
        frame_groups.append(group)

    assert dwg is not None
    del dwg.elements[-1]
    total_seconds = frame_duration_seconds * len(frame_groups)

    style = (
        f".frame{{visibility:hidden; animation:{total_seconds}s linear _k infinite;}}"
    )
    style += f"@keyframes _k{{0%,{100/len(frame_groups)}%{{visibility:visible}}{100/len(frame_groups) * 1.000001}%,100%{{visibility:hidden}}}}"

    for i, group in enumerate(frame_groups):
        dwg.add(group)
        style += f"#{group['id']}{{animation-delay:{i * frame_duration_seconds}s}}"
    dwg.defs.add(svgwrite.container.Style(content=style))
    dwg.saveas(filename)
