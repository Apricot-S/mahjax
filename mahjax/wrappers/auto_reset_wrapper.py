from typing import Optional

import jax
import jax.numpy as jnp

from mahjax._src.types import Array, PRNGKey
from mahjax.core import State

FALSE = jnp.bool_(False)


def auto_reset(step_fn, init_fn):
    """Auto reset wrapper.
    """

    def wrapped_step_fn(state: State, action: Array, key: Optional[PRNGKey] = None):
        assert key is not None, (
        )

        key1, key2 = jax.random.split(key)
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(  # type: ignore
                terminated=FALSE,
                truncated=FALSE,
                rewards=jnp.zeros_like(state.rewards),
            ),
            lambda: state,
        )
        state = step_fn(state, action, key1)
        init_state = init_fn(key2)
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            # state is replaced by initial state,
            # but preserve (terminated, truncated, reward)
            lambda: init_state.replace(  # type: ignore
                terminated=state.terminated,
                truncated=state.truncated,
                rewards=state.rewards,
            ),
            lambda: state,
        )
        return state

    return wrapped_step_fn
