#!/usr/bin/env python3
"""
Behavior Cloning trainer for MahjongAgent.
Uses only the Actor head.
Includes Train/Validation split, logging, and 1-game visualization.
"""
import os
import sys
import pickle
import time
from dataclasses import dataclass
from functools import partial
from typing import Literal
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from tqdm import tqdm

from mahjax._src.visualizer import save_svg_animation
from mahjax.no_red_mahjong.env import Mahjong
from mahjax.no_red_mahjong.rule_based_players import rule_based_player

from mahjax.agents.nash_pg.agent import MahjongAgent, MahjongNetwork

@dataclass
class TrainConfig:
    dataset_path: str = "mahjong_offline_data.pkl"
    batch_size: int = 512
    lr: float = 3e-4
    num_epochs: int = 10
    seed: int = 42
    val_split: float = 0.1 # 10% for validation
    critic_type: Literal["value", "q"] = "value"
    
    # Visualization
    viz_out_dir: str = "fig"
    viz_filename: str = "bc_agent_game.svg"
    viz_max_steps: int = 1000

# cli
conf_dict = OmegaConf.from_cli()
cfg = TrainConfig(**conf_dict)
print(cfg)

# --- Train State ---
class AgentTrainState(TrainState):
    pass

def create_train_state(rng, model, dummy_obs, lr):
    params = model.init(rng, dummy_obs)
    tx = optax.adamw(lr)
    return AgentTrainState.create(apply_fn=model.apply, params=params, tx=tx)

# --- Step Functions ---
@jax.jit
def train_step(state: AgentTrainState, batch):
    """
    BC Update: Maximize log P(expert_action | obs)
    Returns: new_state, loss, accuracy
    """
    obs = batch['obs']
    act = batch['act']
    mask = batch['mask']
    
    def loss_fn(params):
        logits = state.apply_fn(params, obs, method=MahjongNetwork(critic_type=cfg.critic_type).get_action_logits)
        logits = jnp.where(mask, logits, -1e9)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, act).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    
    pred_act = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(pred_act == act)
    
    return new_state, loss, acc

@jax.jit
def eval_step(state: AgentTrainState, batch):
    """
    Validation Step: Compute Loss & Accuracy without update
    Returns: loss, accuracy
    """
    obs = batch['obs']
    act = batch['act']
    mask = batch['mask']
    
    logits = state.apply_fn(state.params, obs, method=MahjongNetwork(critic_type=cfg.critic_type).get_action_logits)
    logits = jnp.where(mask, logits, -1e9)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, act).mean()
    
    pred_act = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(pred_act == act)
    
    return loss, acc

# --- Visualization ---
def make_policy_fn(state: AgentTrainState):
    @jax.jit
    def policy(obs, mask, rng):
        logits = state.apply_fn(state.params, obs, method=MahjongNetwork(critic_type=cfg.critic_type).get_action_logits)
        logits = jnp.where(mask, logits, -1e9)
        return jnp.argmax(logits, axis=-1)
    return policy

def visualize_game(cfg, train_state):
    print("\n=== Visualizing Agent vs Rule-based ===", flush=True)
    env = Mahjong(one_round=True, observe_type="dict")
    jitted_step = jax.jit(env.step)
    policy_fn = make_policy_fn(train_state)
    
    rng = jax.random.PRNGKey(cfg.seed + 999)
    state = env.init(rng)
    history = [state]
    agent_seat = state.current_player
    print(f"Agent is Player {agent_seat}", flush=True)
    
    step = 0
    while not state.terminated and step < cfg.viz_max_steps:
        rng, k_act, k_rule = jax.random.split(rng, 3)
        if state.current_player == agent_seat:
            obs = env.observe(state)
            obs_batched = jax.tree_map(lambda x: x[None, ...], obs)
            mask_batched = state.legal_action_mask[None, ...]
            action = policy_fn(obs_batched, mask_batched, k_act)[0]
        else:
            action = rule_based_player(state, k_rule)
        state = jitted_step(state, action)
        history.append(state)
        step += 1
        if step % 100 == 0: print(f"Step {step}...", flush=True)
            
    print(f"Game End. Score: {state._score}", flush=True)
    os.makedirs(cfg.viz_out_dir, exist_ok=True)
    save_path = os.path.join(cfg.viz_out_dir, cfg.viz_filename)
    save_svg_animation(history, save_path, frame_duration_seconds=0.5)
    print(f"Saved animation to {save_path}", flush=True)

# --- Main ---
def main():
    print("=== Starting BC Training (with Validation) ===", flush=True)
    
    # 1. Load Data
    if not os.path.exists(cfg.dataset_path):
        print(f"Dataset not found: {cfg.dataset_path}", flush=True)
        return

    print("Loading dataset...", flush=True)
    with open(cfg.dataset_path, "rb") as f:
        data = pickle.load(f)
    
    obs_data = data['observation']
    act_data = data['action']
    mask_data = data['legal_action_mask']
    
    num_samples = act_data.shape[0]
    print(f"Loaded {num_samples} samples.", flush=True)
    
    # 2. Train/Val Split
    rng_np = np.random.RandomState(cfg.seed)
    indices = np.arange(num_samples)
    rng_np.shuffle(indices)
    
    split_idx = int(num_samples * (1 - cfg.val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}", flush=True)
    
    # 3. Init Model
    model = MahjongNetwork(critic_type=cfg.critic_type)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)
    
    dummy_obs = jax.tree_map(lambda x: x[0:1], obs_data)
    train_state = create_train_state(init_rng, model, dummy_obs, cfg.lr)
    print("Model initialized.", flush=True)
    
    # 4. Training Loop
    steps_per_epoch = len(train_indices) // cfg.batch_size
    val_steps = len(val_indices) // cfg.batch_size
    if val_steps == 0: val_steps = 1
    
    for epoch in range(cfg.num_epochs):
        # --- Training ---
        np.random.shuffle(train_indices)
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1} [Train]", mininterval=10.0)
        for i in pbar:
            batch_idx = train_indices[i*cfg.batch_size : (i+1)*cfg.batch_size]
            batch = {
                'obs': jax.tree_map(lambda x: x[batch_idx], obs_data),
                'act': act_data[batch_idx],
                'mask': mask_data[batch_idx]
            }
            train_state, loss, acc = train_step(train_state, batch)
            train_loss_sum += loss
            train_acc_sum += acc
            pbar.set_postfix({"loss": f"{float(loss):.4f}", "acc": f"{float(acc):.4f}"})
            
        avg_train_loss = train_loss_sum / steps_per_epoch
        avg_train_acc = train_acc_sum / steps_per_epoch
        
        # --- Validation ---
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        for i in range(val_steps):
            start = i * cfg.batch_size
            end = min((i + 1) * cfg.batch_size, len(val_indices))
            batch_idx = val_indices[start:end]
            
            batch = {
                'obs': jax.tree_map(lambda x: x[batch_idx], obs_data),
                'act': act_data[batch_idx],
                'mask': mask_data[batch_idx]
            }
            loss, acc = eval_step(train_state, batch)
            val_loss_sum += loss
            val_acc_sum += acc
            
        avg_val_loss = val_loss_sum / val_steps
        avg_val_acc = val_acc_sum / val_steps
        
        print(f"Epoch {epoch+1:02d}: "
              f"Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.4f} | "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc:.4f}", flush=True)

    # 5. Save & Visualize
    save_ckpt = "bc_params.pkl"
    with open(save_ckpt, "wb") as f:
        pickle.dump(train_state.params, f)
    print(f"Saved params to {save_ckpt}", flush=True)

    visualize_game(cfg, train_state)

if __name__ == "__main__":
    main()