#!/usr/bin/env python3
"""Single-file Nash PG trainer for MahJax Mahjong with periodic 1-vs-3 eval (Linen version)."""

import sys
import time
from functools import partial
from typing import Dict, Literal, NamedTuple, Any, Optional

import jax
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from omegaconf import OmegaConf
from pydantic import BaseModel
import wandb
import pickle
import distrax

# 修正済みのLinen版Agentをインポート
from mahjax.wrappers.auto_reset_wrapper import auto_reset
from mahjax.no_red_mahjong.env import Mahjong
from mahjax.no_red_mahjong.state import State as MahjongState
from mahjax.no_red_mahjong.rule_based_players import rule_based_player

from .agent import MahjongAgent
from .bc import visualize_game


# All observations emitted by the Mahjong env are nested dicts of arrays.
Observation = Dict[str, jnp.ndarray]
EnvState = MahjongState

# 報酬正規化のための定数 (32000点 = 役満相当で割るなど、320.0指定)
MAX_REWARD = 320.0

NEG = -1e9


# -----------------------------------------------------------------------------
# Configuration utilities
# -----------------------------------------------------------------------------
class NashPGargs(BaseModel):
    env_name: str = "mahjax/no_red_mahjong"
    one_round: bool = True
    seed: int = 0
    lr: float = 3e-4
    ent_coef: float = 0.01  # 少し下げて安定化
    mag_coef: float = 0.2   # Anchorへの引き寄せ強度
    mag_divergence_type: Literal["kl", "l2"] = "kl"
    
    # Magnet (Nash) update frequency
    magnet_update_freq: int = 10000
    
    # ★ 追加: Anchorとして使うパラメータのパス (Agentの初期値には使わない)
    pretrained_path: Optional[str] = "bc_params.pkl"
    
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    num_envs: int = 32
    num_steps: int = 128
    total_timesteps: int = 4e7
    update_epochs: int = 4
    minibatch_size: int = 1024
    gamma: float = 1.0
    gae_lambda: float = 0.95
    wandb_project: str = "mahjax-nashpg-single-file"
    save_model: bool = True
    algo: str = "nashpg_single_file"
    do_eval: bool = True
    eval_interval: int = 50
    eval_num_envs: int = 500
    viz_max_steps: int = 1000

    class args:
        extra = "forbid"


def _parse_args() -> NashPGargs:
    return NashPGargs(**OmegaConf.to_object(OmegaConf.from_cli()))


args = _parse_args()
print(args, file=sys.stderr)

BASE_ENV = Mahjong(one_round=args.one_round, observe_type="dict")
step_fn = auto_reset(BASE_ENV.step, BASE_ENV.init)
NUM_PLAYERS = BASE_ENV.num_players
NUM_UPDATES = int(args.total_timesteps // (args.num_envs * args.num_steps))
if NUM_UPDATES <= 0:
    raise ValueError("total_timesteps must be at least num_envs * num_steps")
BATCH_SIZE = args.num_envs * args.num_steps
if BATCH_SIZE % args.minibatch_size != 0:
    raise ValueError("minibatch_size must divide num_envs * num_steps")
NUM_MINIBATCHES = BATCH_SIZE // args.minibatch_size


# -----------------------------------------------------------------------------
# Data Containers
# -----------------------------------------------------------------------------
TimeStep = NamedTuple(
    "TimeStep",
    [
        ("observation", Observation),
        ("action_mask", jnp.ndarray),
        ("reward", jnp.ndarray),
        ("done", jnp.ndarray),
        ("current_player", jnp.ndarray),
    ],
)
Transition = NamedTuple(
    "Transition",
    [
        ("is_new_eps", jnp.ndarray),
        ("action", jnp.ndarray),
        ("value", jnp.ndarray),
        ("reward", jnp.ndarray),
        ("log_prob", jnp.ndarray),
        ("observation", Observation),
        ("action_mask", jnp.ndarray),
        ("current_player", jnp.ndarray),
    ],
)
LossTerms = NamedTuple(
    "LossTerms",
    [
        ("total_loss", jnp.ndarray),
        ("actor_loss", jnp.ndarray),
        ("critic_loss", jnp.ndarray),
        ("entropy", jnp.ndarray),
        ("approx_kl", jnp.ndarray),
        ("mag_kl", jnp.ndarray),
        ("clip_frac", jnp.ndarray),
        ("explained_var", jnp.ndarray),
    ],
)
RolloutStats = NamedTuple(
    "RolloutStats",
    [
        ("avg_inv_eps", jnp.ndarray),
        ("avg_reward", jnp.ndarray),
    ],
)

class RunnerState(NamedTuple):
    train_state: TrainState
    magnet_params: Any  
    env_state: EnvState
    last_timestep: TimeStep
    rng: jax.random.PRNGKey


# -----------------------------------------------------------------------------
# Generic math helpers
# -----------------------------------------------------------------------------
def masked_mean(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    mask = mask.astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)
    return (x * mask).sum() / denom


def masked_var(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    mean = masked_mean(x, mask)
    mask = mask.astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)
    return ((x - mean) ** 2 * mask).sum() / denom


def masked_std(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(masked_var(x, mask))


# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------
def make_timestep(state: EnvState) -> TimeStep:
    observation = BASE_ENV.observe(state)
    action_mask = state.legal_action_mask.astype(jnp.bool_)
    
    # 報酬のスケーリング (User Request: divide by 320.0)
    reward = jnp.asarray(state.rewards, dtype=jnp.float32) / MAX_REWARD
    
    done = jnp.asarray(state.terminated | state.truncated, dtype=jnp.bool_)
    current_player = jnp.asarray(state.current_player, dtype=jnp.int32)
    return TimeStep(observation, action_mask, reward, done, current_player)


def make_initial_timestep(state: EnvState) -> TimeStep:
    ts = make_timestep(state)
    zero_reward = jnp.zeros_like(ts.reward)
    init_done = jnp.ones_like(ts.done, dtype=jnp.bool_)
    return TimeStep(ts.observation, ts.action_mask, zero_reward, init_done, ts.current_player)


def normalize_advantages(advantages: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    mask = valid_mask.astype(jnp.float32)
    mean = masked_mean(advantages, mask)
    std = masked_std(advantages, mask)
    return (advantages - mean) / (std + 1e-8)


def _flatten(x: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    return x.reshape((batch_size,) + x.shape[2:]) if x.ndim > 2 else x.reshape((batch_size,))


# -----------------------------------------------------------------------------
# Helper to call model methods with params
# -----------------------------------------------------------------------------
def get_action_distribution(model: nn.Module, params, observation, action_mask=None):
    logits = model.apply(params, observation, method=model.get_action_logits)
    if action_mask is not None:
        logits = jnp.where(action_mask, logits, NEG)
    return distrax.Categorical(logits=logits)

def get_action_and_value(model: nn.Module, params, observation, key, action_mask=None):
    logits, value = model.apply(params, observation)
    if action_mask is not None:
        logits = jnp.where(action_mask, logits, NEG)
    dist = distrax.Categorical(logits=logits)
    action, log_prob = dist.sample_and_log_prob(seed=key)
    return action, log_prob, value

def get_value(model: nn.Module, params, observation):
    return model.apply(params, observation, method=model.get_value)


# -----------------------------------------------------------------------------
# Rollout, dataset preparation, and GAE computation
# -----------------------------------------------------------------------------
def collect_and_process(
    agent_model: nn.Module,
    params: Any, 
    env_state: EnvState,
    last_timestep: TimeStep,
    key: jax.random.PRNGKey,
):
    env_state, last_timestep, transitions = collect_trajectories(
        agent_model, params, env_state, last_timestep, key
    )
    advantages, targets, valid_mask = calculate_gae(transitions)
    batch_size = args.num_envs * args.num_steps
    
    flat_transition = Transition(
        is_new_eps=_flatten(transitions.is_new_eps, batch_size),
        action=_flatten(transitions.action, batch_size),
        value=_flatten(transitions.value, batch_size),
        reward=_flatten(transitions.reward, batch_size),
        log_prob=_flatten(transitions.log_prob, batch_size),
        observation=jax.tree.map(lambda x: _flatten(x, batch_size), transitions.observation),
        action_mask=_flatten(transitions.action_mask, batch_size),
        current_player=_flatten(transitions.current_player, batch_size),
    )
    stats = RolloutStats(
        avg_inv_eps=jnp.mean(transitions.is_new_eps.astype(jnp.float32)),
        avg_reward=jnp.mean(transitions.reward[..., 0]),
    )
    flat_advantages = advantages.reshape((batch_size, NUM_PLAYERS))
    flat_targets = targets.reshape((batch_size, NUM_PLAYERS))
    flat_valid_mask = valid_mask.reshape((batch_size, NUM_PLAYERS))
    
    return env_state, last_timestep, flat_transition, flat_advantages, flat_targets, flat_valid_mask, stats


def collect_trajectories(
    agent_model: nn.Module,
    params: Any,
    env_state: EnvState,
    last_timestep: TimeStep,
    key: jax.random.PRNGKey,
):
    def collect_one_env_step(carry, _):
        last_ts, state, rng = carry
        rng, act_key, env_key = jax.random.split(rng, 3)
        batched_ts = jax.tree.map(lambda x: jnp.expand_dims(x, 0), last_ts)
        
        action, log_prob, value = get_action_and_value(
            agent_model, params, batched_ts.observation, act_key, batched_ts.action_mask
        )
        
        action, log_prob, value = (jnp.squeeze(action, 0), jnp.squeeze(log_prob, 0), jnp.squeeze(value, 0))
        last_ts = jax.tree.map(lambda x: jnp.squeeze(x, 0), batched_ts)
        state = step_fn(state, action, env_key)
        new_ts = make_timestep(state)
        transition = Transition(
            is_new_eps=last_ts.done,
            action=action,
            value=value,
            reward=new_ts.reward,
            log_prob=log_prob,
            observation=last_ts.observation,
            action_mask=last_ts.action_mask,
            current_player=last_ts.current_player,
        )
        return (new_ts, state, rng), transition

    def single_env_rollout(carry, _):
        return jax.lax.scan(collect_one_env_step, carry, None, length=args.num_steps)

    batched_rollout = jax.vmap(single_env_rollout, in_axes=(0, None), out_axes=0)
    keys = jax.random.split(key, args.num_envs)
    (last_timestep, env_state, _), transitions = batched_rollout(
        (last_timestep, env_state, keys), None
    )
    return env_state, last_timestep, transitions


def calculate_gae(transitions: Transition):
    def single_env(transitions_single_env: Transition):
        def scan_fn(carry, transition):
            next_gae, next_value, reward_accum, has_next_value, next_is_new, next_valid = carry
            player = transition.current_player
            reward = transition.reward
            value = transition.value
            is_new_eps = transition.is_new_eps
            
            next_gae = jnp.where(next_is_new, jnp.zeros_like(next_gae), next_gae)
            reward_accum = jnp.where(next_is_new, jnp.zeros_like(reward_accum), reward_accum)
            has_next_value = jnp.where(next_is_new, jnp.zeros_like(has_next_value), has_next_value)
            next_value = jnp.where(next_is_new, jnp.zeros_like(next_value), next_value)
            
            reward_accum = reward_accum + reward
            player_reward = reward_accum[player]
            reward_accum = reward_accum.at[player].set(0.0)
            player_has_next = has_next_value[player]
            
            td_error = player_reward + args.gamma * next_value[player] - value
            new_gae_player = td_error + args.gamma * args.gae_lambda * next_gae[player]
            next_gae = next_gae.at[player].set(new_gae_player)
            
            is_valid = player_has_next | next_is_new | next_valid[player]
            
            adv_scalar = jnp.where(is_valid, new_gae_player, 0.0)
            target_scalar = jnp.where(is_valid, adv_scalar + value, value)
            
            advantage = jnp.zeros(NUM_PLAYERS, dtype=jnp.float32).at[player].set(adv_scalar)
            target_value = jnp.zeros(NUM_PLAYERS, dtype=jnp.float32).at[player].set(target_scalar)
            valid_mask = jnp.zeros(NUM_PLAYERS, dtype=jnp.bool_).at[player].set(is_valid)
            
            next_value = next_value.at[player].set(value)
            has_next_value = has_next_value.at[player].set(True)
            next_valid = next_valid.at[player].set(is_valid) | next_is_new
            
            new_carry = (next_gae, next_value, reward_accum, has_next_value, is_new_eps, next_valid)
            return new_carry, (advantage, target_value, valid_mask)

        init = (
            jnp.zeros(NUM_PLAYERS, dtype=jnp.float32),
            jnp.zeros(NUM_PLAYERS, dtype=jnp.float32),
            jnp.zeros(NUM_PLAYERS, dtype=jnp.float32),
            jnp.zeros(NUM_PLAYERS, dtype=jnp.bool_),
            jnp.array(False, dtype=jnp.bool_),
            jnp.zeros(NUM_PLAYERS, dtype=jnp.bool_),
        )
        _, (adv, target, mask) = jax.lax.scan(scan_fn, init, transitions_single_env, reverse=True)
        return adv, target, mask

    return jax.vmap(single_env)(transitions)


# -----------------------------------------------------------------------------
# Training and Evaluation Functions
# -----------------------------------------------------------------------------
def make_update_fn(agent: MahjongAgent):
    agent_model = agent.model

    def loss_terms_for_batch(
            params,
            magnet_params,
            transition_batch: Transition,
            advantage_batch: jnp.ndarray,
            target_batch: jnp.ndarray,
            valid_mask_batch: jnp.ndarray,
        ) -> LossTerms:
            mask = valid_mask_batch.astype(jnp.float32) # (B, 4)
            
            dists = get_action_distribution(agent_model, params, transition_batch.observation, transition_batch.action_mask)
            log_prob = dists.log_prob(transition_batch.action) # (B,)
            log_ratio = log_prob - transition_batch.log_prob   # (B,)
            
            # ratio: (B, 1) -> broadcasts to (B, 4)
            ratio = jnp.exp(log_ratio)[..., None] 
            
            # PPO Loss
            ppo_loss1 = ratio * advantage_batch 
            ppo_loss2 = jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantage_batch
            ppo_loss = -masked_mean(jnp.minimum(ppo_loss1, ppo_loss2), mask)
            
            entropy_val = dists.entropy()[..., None] # (B, 1)
            entropy = masked_mean(entropy_val, mask)
            
            # Magnetic KL regularization
            mag_kl = jnp.array(0.0, dtype=jnp.float32)
            if magnet_params is not None:
                mag_dists = get_action_distribution(agent_model, magnet_params, transition_batch.observation, transition_batch.action_mask)
                if args.mag_divergence_type == "kl":
                    mag_vals = dists.kl_divergence(mag_dists)
                else:
                    probs = dists.probs
                    mag_probs = mag_dists.probs
                    mag_vals = 0.5 * jnp.sum(jnp.square(probs - mag_probs), axis=-1)
                
                mag_kl = masked_mean(mag_vals[..., None], mask)
                
            actor_loss = ppo_loss - args.ent_coef * entropy + args.mag_coef * mag_kl

            # Critic loss
            values = get_value(agent_model, params, transition_batch.observation) # (B,)
            values_expanded = values[..., None] # (B, 1)
            
            value_pred_clipped = transition_batch.value[..., None] + jnp.clip(values_expanded - transition_batch.value[..., None], -args.clip_eps, args.clip_eps)
            
            value_losses = jnp.square(values_expanded - target_batch)
            value_losses_clipped = jnp.square(value_pred_clipped - target_batch)
            
            value_loss = 0.5 * masked_mean(jnp.maximum(value_losses, value_losses_clipped), mask)
            critic_loss = args.vf_coef * value_loss
            
            total_loss = actor_loss + critic_loss

            # Logging
            approx_kl = masked_mean((ratio - 1.0) - log_ratio[..., None], mask)
            clip_frac = masked_mean((jnp.abs(ratio - 1.0) > args.clip_eps).astype(jnp.float32), mask)
            
            target_var = masked_var(target_batch, mask)
            residual_var = masked_var(target_batch - values_expanded, mask)
            explained_var = jnp.maximum(1.0 - residual_var / (target_var + 1e-8), 0.0)
            
            return LossTerms(
                total_loss=total_loss,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                entropy=entropy,
                approx_kl=approx_kl,
                mag_kl=mag_kl,
                clip_frac=clip_frac,
                explained_var=explained_var,
            )

    def _update_step(runner_state: RunnerState, update_idx: int):
        train_state = runner_state.train_state
        magnet_params = runner_state.magnet_params
        env_state = runner_state.env_state
        last_ts = runner_state.last_timestep
        rng = runner_state.rng
        
        rng, collect_key = jax.random.split(rng)
        (
            env_state,
            last_ts,
            batch_transition,
            advantages,
            targets,
            valid_mask,
            stats,
        ) = collect_and_process(
            agent_model,
            train_state.params,
            env_state,
            last_ts,
            collect_key,
        )
        advantages = normalize_advantages(advantages, valid_mask)

        def run_epoch(carry, _):
            ts_state, rng_epoch = carry
            rng_epoch, perm_key = jax.random.split(rng_epoch)
            permutation = jax.random.permutation(perm_key, BATCH_SIZE)
            shuffled = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0),
                (batch_transition, advantages, targets, valid_mask),
            )
            minibatches = jax.tree.map(
                lambda x: x.reshape((NUM_MINIBATCHES, args.minibatch_size) + x.shape[1:]),
                shuffled,
            )

            def train_minibatch(t_state, batch_slice):
                transition_mb, adv_mb, target_mb, mask_mb = batch_slice

                def total_loss_fn(p):
                    return loss_terms_for_batch(
                        p,
                        magnet_params,
                        transition_mb,
                        adv_mb,
                        target_mb,
                        mask_mb,
                    ).total_loss

                grads = jax.grad(total_loss_fn)(t_state.params)
                t_state = t_state.apply_gradients(grads=grads)
                return t_state, None

            ts_state, _ = jax.lax.scan(train_minibatch, ts_state, minibatches)
            return (ts_state, rng_epoch), None

        (train_state, rng), _ = jax.lax.scan(run_epoch, (train_state, rng), None, length=args.update_epochs)
        
        loss_terms = loss_terms_for_batch(
            train_state.params,
            magnet_params,
            batch_transition,
            advantages,
            targets,
            valid_mask,
        )
        
        # Magnet Parameter Update Logic
        should_update = ((update_idx + 1) % args.magnet_update_freq == 0)
        
        new_magnet_params = jax.lax.cond(
            should_update,
            lambda _: train_state.params, 
            lambda _: magnet_params,
            operand=None
        )
        
        runner_state = RunnerState(train_state, new_magnet_params, env_state, last_ts, rng)
        return runner_state, (loss_terms, stats)

    return _update_step


def _random_action(state: EnvState, rng: jax.random.PRNGKey) -> jnp.ndarray:
    logits = jnp.where(state.legal_action_mask, 0.0, NEG)
    return jax.random.categorical(rng, logits=logits)

_rule_based_act = rule_based_player


def make_evaluator(agent_model: nn.Module, num_eval_env: int, baseline_params: Any):
    
    def opponent_action(state: EnvState, rng: jax.random.PRNGKey, opponent_type: int) -> jnp.ndarray:
        return lax.switch(
            opponent_type,
            (
                lambda s, k: _random_action(s, k),
                lambda s, k: get_action_distribution(agent_model, baseline_params, BASE_ENV.observe(s), s.legal_action_mask).mode().reshape(-1)[0],
            ),
            state,
            rng,
        )

    EVAL_STEP_LIMIT = 200

    # --- JIT対象の内部関数 ---
    def run_episode(state, seat_policy_ids, params, opponent_type, key):
        
        def agent_act(s, k):
            obs = BASE_ENV.observe(s)
            mask = jnp.expand_dims(s.legal_action_mask, 0).astype(jnp.bool_)
            dist = get_action_distribution(agent_model, params, obs, mask)
            # Greedy
            return jnp.asarray(dist.mode()).reshape(-1)[0]
        
        def body_fn(carry, _):
            s, rng = carry
            rng, agent_key, opp_key, step_key = jax.random.split(rng, 4)

            def play_step(state_in):
                is_agent = seat_policy_ids[state_in.current_player] == 0
                action = lax.cond(
                    is_agent,
                    lambda _: agent_act(state_in, agent_key),
                    lambda _: opponent_action(state_in, opp_key, opponent_type),
                    operand=None,
                )
                return BASE_ENV.step(state_in, action, step_key)

            next_state = lax.cond(
                s.terminated | s.truncated,
                lambda st: st,
                lambda st: play_step(st),
                operand=s,
            )
            return (next_state, rng), None

        (final_state, _), _ = lax.scan(
            body_fn,
            (state, key),
            None,
            length=EVAL_STEP_LIMIT,
        )
        return final_state

    # num_eval_env 分を一括で vmap
    vmapped_episode = jax.vmap(run_episode, in_axes=(0, 0, None, None, 0))

    def compute_metrics(final_states, seat_policy_ids):
        # 変数準備: (N, 4)
        scores = final_states._score
        is_agent = (seat_policy_ids == 0)
        
        # --- 1. Score Stats ---
        agent_scores = (scores * is_agent).sum(axis=1) # (N,)
        opp_scores_sum = (scores * (~is_agent)).sum(axis=1)
        opp_scores_avg = opp_scores_sum / 3.0
        
        # --- 2. Rank (平均着順) ---
        better_count = (scores > agent_scores[:, None]).sum(axis=1)
        equal_count = (scores == agent_scores[:, None]).sum(axis=1)
        ranks = 1.0 + better_count + (equal_count - 1.0) * 0.5
        
        # --- 3. Win Rate (Strict Top) ---
        game_win = (agent_scores > (scores * (~is_agent)).max(axis=1)).astype(jnp.float32)
        
        # --- 4. Mahjong Metrics ---
        hora = (final_states._has_won & is_agent).any(axis=1).astype(jnp.float32)
        riichi = (final_states._riichi & is_agent).any(axis=1).astype(jnp.float32)
        meld = ((final_states._n_meld > 0) & is_agent).any(axis=1).astype(jnp.float32)

        # 平均・最大・最小をここで計算して返す
        # Note: すべて JAX Scalars として返す
        return {
            "win_rate": game_win.mean(),
            "avg_margin": (agent_scores - opp_scores_avg).mean(),
            "agent_score": agent_scores.mean(),
            "opponent_score": opp_scores_avg.mean(),
            
            "avg_rank": ranks.mean(),
            "hora_rate": hora.mean(),
            "riichi_rate": riichi.mean(),
            "meld_rate": meld.mean(),
            "score_max": agent_scores.max(),
            "score_min": agent_scores.min(),
        }

    # Python関数 evaluate は内部で JIT された関数を呼ぶ
    # JIT対象は "1回の評価実行" の部分だけ
    def evaluate(params, key: jax.random.PRNGKey):
        
        def play_all(k, opponent_type):
            keys = jax.random.split(k, num_eval_env * 3)
            keys_init = keys[:num_eval_env]
            keys_assign = keys[num_eval_env : 2 * num_eval_env]
            keys_play = keys[2 * num_eval_env :]
            
            solo_seat = jax.vmap(lambda k: jax.random.randint(k, (), 0, NUM_PLAYERS))(keys_assign)
            seats = jnp.arange(NUM_PLAYERS)[None, :]
            seat_policy_ids = jnp.where(seats == solo_seat[:, None], 0, 1).astype(jnp.int32)
            
            init_states = jax.vmap(BASE_ENV.init)(keys_init)
            final_states = vmapped_episode(init_states, seat_policy_ids, params, opponent_type, keys_play)
            return compute_metrics(final_states, seat_policy_ids)

        key, rand_key, base_key = jax.random.split(key, 3)
        
        # JIT実行 (返り値は JAX Tracers/Arrays)
        rand_metrics = play_all(rand_key, 0)
        base_metrics = play_all(base_key, 1)
    
        
        eval_log = {
            "vs_rand/win_rate": rand_metrics["win_rate"],
            "vs_rand/avg_margin": rand_metrics["avg_margin"],
            "vs_rand/agent_score": rand_metrics["agent_score"],
            "vs_rand/opponent_score": rand_metrics["opponent_score"],
            "vs_rand/avg_rank": rand_metrics["avg_rank"],
            "vs_rand/hora_rate": rand_metrics["hora_rate"],
            "vs_rand/riichi_rate": rand_metrics["riichi_rate"],
            "vs_rand/meld_rate": rand_metrics["meld_rate"],
            "vs_rand/score_max": rand_metrics["score_max"],
            "vs_rand/score_min": rand_metrics["score_min"],
            
            "vs_baseline/win_rate": base_metrics["win_rate"],
            "vs_baseline/avg_margin": base_metrics["avg_margin"],
            "vs_baseline/agent_score": base_metrics["agent_score"],
            "vs_baseline/opponent_score": base_metrics["opponent_score"],
            "vs_baseline/avg_rank": base_metrics["avg_rank"],
            "vs_baseline/hora_rate": base_metrics["hora_rate"],
            "vs_baseline/riichi_rate": base_metrics["riichi_rate"],
            "vs_baseline/meld_rate": base_metrics["meld_rate"],
            "vs_baseline/score_max": base_metrics["score_max"],
            "vs_baseline/score_min": base_metrics["score_min"],
        }
        return key, eval_log

    return evaluate
# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------
def train(rng_key: jax.random.PRNGKey):
    rng, agent_key = jax.random.split(rng_key)
    
    # 1. Initialize Random Policy (Always random at start)
    agent = MahjongAgent(agent_key, critic_type="value")
    tx = optax.adamw(args.lr, eps=1e-5)
    train_state = TrainState.create(
        apply_fn=agent.model.apply,
        params=agent.params,
        tx=tx,
    )
    
    # 2. Determine Initial Magnet Params (Anchor)
    # If path provided, load it. Else, create copy of random params.
    if args.pretrained_path:
        print(f"Loading anchor parameters from: {args.pretrained_path}", flush=True)
        with open(args.pretrained_path, "rb") as f:
            loaded = pickle.load(f)
            
            # 構造チェックと整形
            # Flaxのapplyは {'params': ...} という形式を要求する
            if isinstance(loaded, dict) and "params" in loaded:
                # 既に {'params': ...} の形ならそのまま使う
                magnet_params = loaded
            else:
                # 生のパラメータ木の場合、'params' キーでラップする
                magnet_params = {"params": loaded}
    else:
        print("No pretrained path provided. Using random init as anchor.", flush=True)
        magnet_params = agent.params

    rng, reset_key = jax.random.split(rng)
    env_state = jax.vmap(BASE_ENV.init)(jax.random.split(reset_key, args.num_envs))
    init_ts = jax.vmap(make_initial_timestep)(env_state)

    rng, eval_rng = jax.random.split(rng)
    runner_state = RunnerState(train_state, magnet_params, env_state, init_ts, rng)
    
    # Build JIT functions
    update_step = jax.jit(make_update_fn(agent))
    evaluator = jax.jit(make_evaluator(agent.model, args.eval_num_envs, magnet_params))

    steps = 0
    start_time = time.time()

    def _eval_and_log(eval_rng, steps, update_idx, current_params):
        eval_rng, eval_log = evaluator(current_params, eval_rng)
        eval_log = {k: float(v) for k, v in eval_log.items()}
        wandb.log({"steps": steps, "update": update_idx, **eval_log})
        print({"steps": steps, "update": update_idx, **eval_log}, flush=True)
        return eval_rng

    if args.do_eval:
        eval_rng = _eval_and_log(eval_rng, steps, 0, runner_state.train_state.params)

    for update_idx in range(NUM_UPDATES):
        runner_state, (loss_terms, stats) = update_step(runner_state, jnp.array(update_idx, dtype=jnp.int32))
        
        steps += args.num_envs * args.num_steps
        avg_inv_eps = stats.avg_inv_eps
        avg_reward = stats.avg_reward
        eps_len = jnp.where(avg_inv_eps > 0, 1.0 / avg_inv_eps, jnp.nan)
        avg_return = jnp.where(avg_inv_eps > 0, avg_reward / avg_inv_eps, jnp.nan)
        
        log = {
            "steps": steps,
            "update": update_idx + 1,
            f"train/loss_total": float(loss_terms.total_loss),
            f"train/loss_actor": float(loss_terms.actor_loss),
            f"train/loss_critic": float(loss_terms.critic_loss),
            f"train/entropy": float(loss_terms.entropy),
            f"train/approx_kl": float(loss_terms.approx_kl),
            f"train/mag_kl": float(loss_terms.mag_kl),
            f"train/clip_frac": float(loss_terms.clip_frac),
            f"train/explained_var": float(loss_terms.explained_var),
            f"train/avg_reward_step": float(avg_reward),
            f"train/avg_eps_len": float(eps_len),
            f"train/avg_return": float(avg_return),
        }
        wandb.log(log)


        if args.do_eval and ((update_idx + 1) % args.eval_interval == 0 or update_idx + 1 == NUM_UPDATES):
            eval_rng = _eval_and_log(eval_rng, steps, update_idx + 1, runner_state.train_state.params)
            #print(log, flush=True)

    wandb.log({"train_time": time.time() - start_time, "steps": steps})
    return runner_state.train_state


if __name__ == "__main__":
    wandb.init(project=args.wandb_project, config=args.dict())
    rng = jax.random.PRNGKey(args.seed)
    final_train_state = train(rng)
    if args.save_model:
        # Save params via flax serialization or orbax
        with open(f"{args.env_name}-seed={args.seed}.ckpt", "wb") as fh:
            pickle.dump(final_train_state.params, fh)

    visualize_game(args, final_train_state)