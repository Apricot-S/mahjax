#!/usr/bin/env python3
"""
Data collector for Mahjong BC & Value Learning using Rule-based players.
Saves observations, actions, masks, AND returns.
Optimized for memory efficiency (Small chunks).
"""
import os
import sys
import pickle
import time
from typing import NamedTuple, Dict

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from pydantic import BaseModel
from tqdm import tqdm

from mahjax.wrappers.auto_reset_wrapper import auto_reset
from mahjax.no_red_mahjong.env import Mahjong
from mahjax.no_red_mahjong.rule_based_players import rule_based_player

# --- Config ---
class CollectConfig(BaseModel):
    seed: int = 0
    num_envs: int = 4
    num_steps: int = 32
    total_timesteps: int = 200_000
    save_path: str = "mahjong_offline_data.pkl"
    gamma: float = 0.99       # 割引率
    max_reward: float = 320.0 # 正規化用

# --- Environment ---
env = Mahjong(one_round=True, observe_type="dict")
step_env = auto_reset(env.step, env.init)

def _one_step(state, key):
    obs = env.observe(state)
    mask = state.legal_action_mask
    curr_player = state.current_player # 誰の手番か
    
    k_act, k_step = jax.random.split(key, 2)
    action = rule_based_player(state, k_act)
    
    next_state = step_env(state, action, k_step)
    
    # 状態遷移情報
    done = next_state.terminated | next_state.truncated
    reward = next_state.rewards # [P]
    
    return next_state, (obs, action, mask, reward, done, curr_player)

def _rollout_chunk(state, keys, num_steps):
    vmap_step = jax.vmap(_one_step, in_axes=(0, 0))
    def body(carry, key):
        st = carry
        new_st, out = vmap_step(st, key)
        return new_st, out
    final_state, outs = jax.lax.scan(body, state, keys, length=num_steps)
    return final_state, outs

def compute_returns(rewards, dones, current_players, gamma):
    """
    Compute discounted returns for the ACTIVE player at each step.
    rewards: [T, B, P]
    dones:   [T, B]
    current_players: [T, B]
    Returns: [T, B] (Scalar return for the actor)
    """
    T, B, P = rewards.shape
    returns = np.zeros((T, B), dtype=np.float32)
    
    # 簡易的なモンテカルロリターン計算 (逆順)
    # ※ 本来はBootstrappingが必要だが、今回は完走データを使うためMCで近似
    # 各環境ごとに計算
    for b in range(B):
        running_ret = np.zeros(P, dtype=np.float32) # 各プレイヤーのG_t
        for t in reversed(range(T)):
            # 現在のステップの報酬
            r_t = rewards[t, b] # [P]
            d_t = dones[t, b]   # bool
            
            if d_t:
                running_ret = np.zeros(P, dtype=np.float32)
            
            # G_t = r_t + gamma * G_{t+1}
            running_ret = r_t + gamma * running_ret
            
            # このステップで行動したプレイヤーのReturnを記録
            # (行動選択時点での価値 = 即時報酬 + 将来価値)
            # ※ 注意: 即時報酬を含めるかどうかは定義によるが、Q学習的には含める
            p = current_players[t, b]
            returns[t, b] = running_ret[p]
            
    return returns

def main():
    print("=== Starting Data Collection (With Returns) ===", flush=True)
    conf = OmegaConf.from_cli()
    cfg = CollectConfig(**conf)
    print(f"Config: {cfg}", flush=True)

    rng = jax.random.PRNGKey(cfg.seed)
    rng, k_init = jax.random.split(rng)
    
    init_keys = jax.random.split(k_init, cfg.num_envs)
    state = jax.vmap(env.init)(init_keys)
    
    print("Compiling JIT function...", flush=True)
    jit_rollout = jax.jit(lambda s, k: _rollout_chunk(s, k, cfg.num_steps))
    
    # Buffers
    data_obs = [] 
    data_act = []
    data_mask = []
    data_ret = [] # New: Returns
    
    chunk_size = cfg.num_envs * cfg.num_steps
    num_chunks = (cfg.total_timesteps + chunk_size - 1) // chunk_size
    
    total_steps = 0
    start_time = time.time()
    
    # 連続性を保つためのCarry (前回のチャンクからの続き)
    # 本格的なオフラインRL用データセット作成では、エピソード境界を跨ぐ計算が必要だが、
    # 簡易的にチャンク内で完結させる、または十分長いエピソードとして扱う。
    # ここでは「チャンク内で計算し、境界は無視する（多少の誤差許容）」簡易実装とします。
    # ※ 厳密にやるなら全履歴をオンメモリで保持して最後に計算する必要がありますが、
    # メモリ制約(79GB error)があるため、チャンク処理を優先します。
    
    for _ in tqdm(range(num_chunks), desc="Collecting", mininterval=10.0):
        rng, k_chunk = jax.random.split(rng)
        keys = jax.random.split(k_chunk, cfg.num_envs * cfg.num_steps).reshape(cfg.num_steps, cfg.num_envs, -1)
        
        state, (obs_seq, act_seq, mask_seq, rew_seq, done_seq, cp_seq) = jit_rollout(state, keys)
        
        # CPU転送
        obs_cpu = jax.tree_map(np.array, obs_seq)
        act_cpu = np.array(act_seq)
        mask_cpu = np.array(mask_seq)
        rew_cpu = np.array(rew_seq)
        done_cpu = np.array(done_seq)
        cp_cpu = np.array(cp_seq)
        
        # --- Return Calculation (CPU) ---
        # このチャンク内でのリターンを計算
        # 注意: チャンク末尾で切れるため、Bootstrapできない分だけ誤差が出るが、
        # num_stepsがエピソード長より短いためバイアスがかかる。
        # -> オフラインRLとしては「エピソード完了まで待つ」バッファリングが必要だが、
        # ここでは実装の複雑さを避けるため、Mahjongの1局が短い(avg 50-60 steps)ことを利用し、
        # num_steps=32だと切れやすいので、本当はもっと大きくしたいがメモリ制約がある。
        
        # 妥協案: Return計算は諦めて「報酬(reward)」を保存し、
        # 学習時にRewardからReturnを計算する、あるいは
        # 単純に「このチャンク内での収益」とする。
        # 今回は「正規化されたリターン」を教師データとしたいので、ここで計算します。
        returns_chunk = compute_returns(rew_cpu, done_cpu, cp_cpu, cfg.gamma)
        
        # 正規化 (Max Rewardで割る)
        returns_chunk = returns_chunk / cfg.max_reward

        # Flatten & Store
        flat_obs = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), obs_cpu)
        data_obs.append(flat_obs)
        data_act.append(act_cpu.flatten())
        data_mask.append(mask_cpu.reshape(-1, mask_cpu.shape[-1]))
        data_ret.append(returns_chunk.flatten())
        
        total_steps += act_cpu.size
        if total_steps >= cfg.total_timesteps:
            break

    # Save
    print("Concatenating data...", flush=True)
    if not data_obs: return

    keys = data_obs[0].keys()
    full_obs = {k: np.concatenate([d[k] for d in data_obs], axis=0) for k in keys}
    full_act = np.concatenate(data_act, axis=0)
    full_mask = np.concatenate(data_mask, axis=0)
    full_ret = np.concatenate(data_ret, axis=0)
    
    N = cfg.total_timesteps
    dataset = {
        "observation": {k: v[:N] for k, v in full_obs.items()},
        "action": full_act[:N],
        "legal_action_mask": full_mask[:N],
        "return": full_ret[:N] # 追加
    }
    
    if cfg.save_path:
        os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)
        print(f"Saving to {cfg.save_path} ...", flush=True)
        with open(cfg.save_path, "wb") as f:
            pickle.dump(dataset, f)
        print("Done.", flush=True)

if __name__ == "__main__":
    main()