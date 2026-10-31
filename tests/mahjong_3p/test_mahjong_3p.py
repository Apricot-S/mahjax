"""The 3-player table: the wall, kita, payments and the env, as Tenhou's 三人麻雀 plays them."""

import functools

import jax
import jax.numpy as jnp

import mahjax
from mahjax.common.action import Action
from mahjax.common.hand import Hand
from mahjax.common.logic import (
    _is_uninterrupted_first_turn,
    _mangan_tsumo,
    _mask_after_draw,
    _mask_after_draw_in_riichi,
    _noten_payments,
    _replace_state,
    _tsumo_payments,
)
from mahjax.common.tile import Tile
from mahjax.common.yaku import Yaku, _dora_array_from_state
from mahjax.mahjong_3p.config import NO_RED, RED
from mahjax.mahjong_3p.constants import FIRST_DRAW_IDX, KITA_ACTION, NORTH
from mahjax.mahjong_3p.env import Mahjong3p
from mahjax.mahjong_3p.logic import _kita
from mahjax.mahjong_3p.state import State

STEP_KEY = jax.random.PRNGKey(0)


@functools.lru_cache(maxsize=None)
def _env(round_end="auto", game_config=RED):
    env = Mahjong3p(round_end=round_end, game_config=game_config)
    return env, jax.jit(env.init), jax.jit(env.step)


def _hand(tile_types):
    hand = jnp.zeros((Tile.NUM_TILE_TYPE_WITH_RED,), dtype=jnp.int8)
    for tile in tile_types:
        hand = hand.at[tile].add(1)
    return hand


def test_make_builds_the_3p_presets() -> None:
    assert {"red_3p", "no_red_3p"} <= set(mahjax.available_envs())
    red, no_red = mahjax.make("red_3p"), mahjax.make("no_red_3p")
    assert red.game_config is RED and red.id == "red_3p"
    assert no_red.game_config is NO_RED and no_red.id == "no_red_3p"
    assert red.num_players == 3 and red.num_actions == 88
    assert Mahjong3p(round_mode="east").round_limit == 2  # East 1-3
    assert Mahjong3p(round_mode="half").round_limit == 5  # East 1-3, South 1-3


def test_the_wall_has_no_2m_to_8m_and_a_red_5p_and_5s() -> None:
    _, init, _ = _env()
    state = init(jax.random.PRNGKey(1))
    deck = state.round_state.deck
    assert deck.shape == (108,)
    types = Tile.to_tile_type(deck)
    assert not bool(((types >= 1) & (types <= 7)).any())
    assert sorted(int(t) for t in deck if int(t) >= Tile.NUM_TILE_TYPE) == [Tile.RED_FIVE["p"], Tile.RED_FIVE["s"]]
    _, init_no_red, _ = _env(game_config=NO_RED)
    assert int(init_no_red(jax.random.PRNGKey(1)).round_state.deck.max()) < Tile.NUM_TILE_TYPE


def test_three_hands_are_dealt_at_35000() -> None:
    _, init, _ = _env()
    state = init(jax.random.PRNGKey(2))
    dealer = int(state.round_state.dealer)
    sizes = state.players.hand.sum(axis=1).tolist()
    assert sizes[dealer] == 14 and sorted(sizes) == [13, 13, 14]
    assert state.round_state.score.tolist() == [350, 350, 350]
    assert int(state.round_state.next_deck_ix) == FIRST_DRAW_IDX - 1


def test_a_1m_indicator_makes_9m_the_dora() -> None:
    state = State()
    state = _replace_state(state, dora_indicators=state.round_state.dora_indicators.at[0].set(0))
    dora = _dora_array_from_state(state)[0]
    assert int(dora[8]) == 1 and int(dora.sum()) == 1


def test_kita_sets_a_north_aside_and_draws_a_replacement() -> None:
    _, init, step = _env()
    for seed in range(100):
        state = init(jax.random.PRNGKey(seed))
        if bool(state.legal_action_mask[KITA_ACTION]):
            break
    c_p = int(state.current_player)
    out = step(state, jnp.int32(KITA_ACTION), STEP_KEY)
    if not bool(out.legal_action_mask[Action.RON]):  # nobody could ron the north
        assert int(out.current_player) == c_p
        assert int(out.players.kita[c_p]) == 1
        assert int(out.players.hand[c_p, NORTH]) == int(state.players.hand[c_p, NORTH]) - 1
        assert int(out.players.hand[c_p].sum()) == 14  # the replacement is in hand
        assert int(out.round_state.last_deck_ix) == int(state.round_state.last_deck_ix) + 1
        assert bool(out.round_state.can_after_kan)


def test_kita_is_not_offered_on_the_haitei_draw_and_in_riichi_only_for_a_drawn_north() -> None:
    state = State()
    hand = jnp.zeros((3, Tile.NUM_TILE_TYPE_WITH_RED), dtype=jnp.int8).at[0].set(
        _hand([0, 0, 0, 8, 8, 8, 9, 10, 11, 18, 19, 20, NORTH, 31])
    )
    assert bool(_mask_after_draw(state, hand, jnp.int32(0), jnp.int8(31))[KITA_ACTION])
    haitei = _replace_state(state, is_haitei=jnp.bool_(True))
    assert not bool(_mask_after_draw(haitei, hand, jnp.int32(0), jnp.int8(31))[KITA_ACTION])
    assert not bool(_mask_after_draw_in_riichi(state, hand, jnp.int32(0), jnp.int8(31))[KITA_ACTION])
    assert bool(_mask_after_draw_in_riichi(state, hand, jnp.int32(0), jnp.int8(NORTH))[KITA_ACTION])


def test_each_north_set_aside_is_a_dora() -> None:
    hand = _hand([0, 0, 0, 8, 8, 8, 9, 10, 11, 18, 19, 20, 31])  # white wait
    base = _replace_state(State(), last_draw=jnp.int8(31))

    def fan(kita, indicator=-1):
        state = _replace_state(
            base,
            kita=base.players.kita.at[0].set(kita),
            dora_indicators=base.round_state.dora_indicators.at[0].set(indicator),
        )
        return int(Yaku.judge(hand, jnp.bool_(False), jnp.int8(0), state)[1])

    assert fan(2) == fan(0) + 2
    assert fan(2, indicator=29) == fan(0, indicator=29) + 4  # a west indicator makes north a dora


def _kita_table(can_ron_north: bool, ippatsu: bool = False) -> State:
    """Player 0 is about to set a north aside; player 1, in riichi, waits on north with a yaku (white).
    Riichi and white make 2600, so ippatsu shows in the payment."""
    state = State()
    waiting = _hand([31, 31, 31, 10, 11, 12, 13, 14, 15, 20, 21, 22, NORTH])
    hands = jnp.zeros((3, Tile.NUM_TILE_TYPE_WITH_RED), dtype=jnp.int8)
    hands = hands.at[0].set(_hand([NORTH, 12, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]))
    hands = hands.at[1].set(waiting)
    can_win = jnp.zeros((3, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[1, NORTH].set(can_ron_north)
    return _replace_state(
        state,
        current_player=jnp.int8(0),
        hand=jax.vmap(Hand.to_34)(hands),
        hand_with_red=hands,
        can_win=can_win,
        riichi=state.players.riichi.at[1].set(True),
        ippatsu=state.players.ippatsu.at[1].set(ippatsu),
        next_deck_ix=jnp.int32(50),
        legal_action_mask=state.players.legal_action_mask.at[0, KITA_ACTION].set(True),
    )


def test_the_others_may_ron_a_kita_north_without_chankan() -> None:
    out = _kita(_kita_table(can_ron_north=True))
    assert bool(out.round_state.kita_declared) and not bool(out.round_state.kan_declared)
    assert int(out.current_player) == 1 and bool(out.legal_action_mask[Action.RON])
    assert int(out.round_state.target) == NORTH and int(out.round_state.last_player) == 0


def test_a_ronned_kita_north_ends_the_round_and_keeps_ippatsu() -> None:
    _, _, step = _env(round_end="confirm")

    def ron(ippatsu):
        kitad = step(_kita_table(can_ron_north=True, ippatsu=ippatsu), jnp.int32(KITA_ACTION), STEP_KEY)
        return step(kitad, jnp.int32(Action.RON), STEP_KEY)

    won = ron(ippatsu=True)
    assert bool(won.round_state.terminated_round) and not bool(won.round_state.kita_declared)
    assert int(won.players.hand[0].sum()) == 13  # no replacement for a ronned north
    assert jnp.flatnonzero(won.legal_action_mask).tolist() == [Action.DUMMY]
    assert float(won.rewards[1]) > float(ron(ippatsu=False).rewards[1])


def test_a_kita_nobody_rons_ends_every_ippatsu() -> None:
    _, _, step = _env(round_end="confirm")
    out = step(_kita_table(can_ron_north=False, ippatsu=True), jnp.int32(KITA_ACTION), STEP_KEY)
    assert int(out.current_player) == 0 and int(out.players.hand[0].sum()) == 14
    assert not bool(out.players.ippatsu.any())


def test_a_kita_interrupts_the_first_go_around() -> None:
    state = State()
    assert bool(_is_uninterrupted_first_turn(state, jnp.int32(FIRST_DRAW_IDX - 1)))
    kitad = _replace_state(state, kita=state.players.kita.at[2].set(1))
    assert not bool(_is_uninterrupted_first_turn(kitad, jnp.int32(FIRST_DRAW_IDX - 1)))


def test_tsumo_son_and_honba_and_noten() -> None:
    # A non-dealer's mangan tsumo: 2000 from the other non-dealer, 4000 from the dealer.
    assert _tsumo_payments(jnp.int32(1), jnp.int32(0), 2000, jnp.int8(0), 3).tolist() == [-40, 60, -20]
    # The dealer's: 4000 from each.
    assert _tsumo_payments(jnp.int32(0), jnp.int32(0), 2000, jnp.int8(0), 3).tolist() == [80, -40, -40]
    # Two honba: 100 more from each payer per honba.
    assert _mangan_tsumo(State(), jnp.int32(1), jnp.int32(0), jnp.int8(2)).tolist() == [-42, 64, -22]
    # The noten players pay 2000 between them.
    state = State()
    assert _noten_payments(state, jnp.array([True, False, False])).tolist() == [20, -10, -10]
    assert _noten_payments(state, jnp.array([True, True, False])).tolist() == [10, 10, -20]
    assert _noten_payments(state, jnp.array([True, True, True])).tolist() == [0, 0, 0]


def test_the_observation_has_three_seats_and_the_kita() -> None:
    env, init, _ = _env()
    obs = env.observe(init(jax.random.PRNGKey(3)))
    assert obs["scores"].shape == (3,) and obs["riichi"].shape == (3,)
    assert obs["kita"].shape == (3,)


def test_random_games_end_with_the_points_intact() -> None:
    env, _, _ = _env()

    @jax.jit
    def play(key):
        def body(state, k):
            k1, k2 = jax.random.split(k)
            action = jax.random.categorical(k1, jnp.where(state.legal_action_mask, 0.0, -jnp.inf))
            return env.step(state, action, k2), None

        state, _ = jax.lax.scan(body, env.init(key), jax.random.split(key, 1500))
        return state

    state = jax.vmap(play)(jax.random.split(jax.random.PRNGKey(4), 2))
    assert bool(state.terminated.all())
    assert state.round_state.score.sum(axis=1).tolist() == [1050] * 2


def test_the_rule_based_player_plays_legally_and_sets_every_north_aside() -> None:
    env, _, _ = _env()

    @jax.jit
    def play(key):
        def body(carry, k):
            state, illegal, missed, kitas = carry
            k1, k2 = jax.random.split(k)
            action = mahjax.rule_based_player(state, k1)
            live, mask = ~state.terminated, state.legal_action_mask
            illegal = illegal | (live & ~mask[action])
            # A north goes aside whenever it may, unless the hand can win.
            missed = missed | (live & mask[KITA_ACTION] & ~mask[Action.TSUMO] & (action != KITA_ACTION))
            kitas = kitas + (live & (action == KITA_ACTION))
            return (env.step(state, action, k2), illegal, missed, kitas), None

        carry = (env.init(key), jnp.bool_(False), jnp.bool_(False), jnp.int32(0))
        (state, illegal, missed, kitas), _ = jax.lax.scan(body, carry, jax.random.split(key, 1500))
        return state, illegal, missed, kitas

    state, illegal, missed, kitas = jax.vmap(play)(jax.random.split(jax.random.PRNGKey(5), 2))
    assert bool(state.terminated.all())
    assert not bool(illegal.any()) and not bool(missed.any())
    assert int(kitas.sum()) > 0
    assert state.round_state.score.sum(axis=1).tolist() == [1050] * 2
