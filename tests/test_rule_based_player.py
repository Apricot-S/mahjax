"""``mahjax.rule_based_player`` keeps working outside a trace after a fresh ``import mahjax``."""

import os
import subprocess
import sys
from pathlib import Path

#: Trace the player once, then call it as it is, as a program does that jits it and
#: later looks at one action. A player module first imported inside that trace
#: would keep the trace's tracers in its constants, and the plain call would trip
#: over them.
SCRIPT = """
import jax

import mahjax
from mahjax.mahjong_3p.state import State as ThreePlayerState
from mahjax.mahjong_4p.state import State as FourPlayerState

key = jax.random.PRNGKey(0)
for state in (FourPlayerState(), ThreePlayerState()):
    jax.jit(mahjax.rule_based_player).lower(state, key)  # traced, not compiled
    mahjax.rule_based_player(state, key)
"""


def test_the_rule_based_player_runs_after_a_trace_from_a_fresh_import() -> None:
    # Its own process: this one imported the players long before.
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root), "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "cpu")}
    result = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
