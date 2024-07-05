import json
import os
import random

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from tic_tac_toe.environment import TicTacToe
from tic_tac_toe.model import DQN
from utils.replay_buffer import PrioritizedReplayBuffer

CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def choose_action(state, policy_net, epsilon, env):
    if random.random() < epsilon:
        return random.choice(env.available_actions())
    else:
        with torch.no_grad():
            state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state)
            available_actions = env.available_actions()
            available_q_values = [q_values[0, action[0] * 3 + action[1]].item() for action in available_actions]
            max_q_value = max(available_q_values)
            max_q_action = available_actions[available_q_values.index(max_q_value)]
            return max_q_action

def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma, beta):
    if len(memory) < batch_size:
        return 0
    states, actions, rewards, next_states, dones, indices, weights = memory.sample(batch_size, beta)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)

    state_action_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_state_actions = policy_net(next_states).max(1)[1].unsqueeze(1)
    next_state_values = target_net(next_states).gather(1, next_state_actions).squeeze(1).detach()
    expected_state_action_values = rewards + (gamma * next_state_values * (1 - dones))

    loss = (state_action_values - expected_state_action_values).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    memory.update_priorities(indices, prios.detach().cpu().numpy())
    
    return loss.item()

def save_checkpoint(policy_net, target_net, optimizer, episode, architecture):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{episode}.pth')
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'architecture': architecture
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(policy_net, target_net, architecture, optimizer=None):
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_')]
    if not checkpoints:
        return 0
    latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[1].split('.')[0]))
    checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    
    if checkpoint.get('architecture', 'default') != architecture:
        print("Skipping checkpoint loading due to architecture mismatch.")
        return 0
    
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode']
    print(f"Checkpoint loaded from {checkpoint_path}, resuming from episode {episode}")
    return episode + 1

def reinitialize_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def train_dqn(env, policy_net, target_net, num_episodes=5000, batch_size=128, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500, target_update=50, save_interval=1000, architecture='default', update_checkpoints=True, alpha=0.6, beta_start=0.4):
    optimizer = optim.Adam(policy_net.parameters())
    memory = PrioritizedReplayBuffer(10000, alpha)

    start_episode = load_checkpoint(policy_net, target_net, architecture, optimizer) if update_checkpoints else 0

    total_loss = 0

    for episode in range(start_episode, num_episodes):
        state = env.reset()
        episode_loss = 0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)
        beta = min(1.0, beta_start + episode * (1.0 - beta_start) / num_episodes)
        for t in range(9):
            action = choose_action(state, policy_net, epsilon, env)
            while True:
                try:
                    next_state, done, winner = env.step(action, 1 if t % 2 == 0 else -1)
                    break
                except ValueError:
                    action = choose_action(state, policy_net, epsilon, env)
            reward = 1 if winner == 1 else -1 if winner == -1 else 0
            memory.push(state.flatten(), action[0] * 3 + action[1], reward, next_state.flatten(), done)
            state = next_state
            if done:
                break
            loss = optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma, beta)
            episode_loss += loss

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % save_interval == 0 and update_checkpoints:
            save_checkpoint(policy_net, target_net, optimizer, episode, architecture)

        if episode % 100 == 0:
            avg_loss = episode_loss / (t + 1)
            print(f"Episode {episode}/{num_episodes}, Loss: {avg_loss:.6f}")

        total_loss += episode_loss

    avg_total_loss = total_loss / num_episodes
    print("Training complete.")
    return avg_total_loss

def objective(trial):
    env = TicTacToe()
    policy_net = DQN(architecture='default')
    target_net = DQN(architecture='default')
    target_net.load_state_dict(policy_net.state_dict())
    reinitialize_weights(policy_net)
    reinitialize_weights(target_net)
    
    num_episodes = trial.suggest_int('num_episodes', 1000, 5000)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.99)
    epsilon_start = trial.suggest_uniform('epsilon_start', 0.8, 1.0)
    epsilon_end = trial.suggest_uniform('epsilon_end', 0.01, 0.1)
    epsilon_decay = trial.suggest_int('epsilon_decay', 100, 1000)
    target_update = trial.suggest_int('target_update', 10, 50)
    save_interval = trial.suggest_int('save_interval', 1000, 5000)

    avg_loss = train_dqn(
        env, policy_net, target_net,
        num_episodes=num_episodes,
        batch_size=batch_size,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update=target_update,
        save_interval=save_interval,
        update_checkpoints=False
    )
    
    return avg_loss

def run_experiments(num_experiments=10, num_episodes=5000):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_experiments, n_jobs=4)  # Using 4 parallel jobs

    best_trial = study.best_trial
    print(f"Best hyperparameters: {best_trial.params} with avg loss: {best_trial.value:.6f}")

    # Convert DataFrame to dict with stringified timestamps and timedeltas
    trials_df = study.trials_dataframe().applymap(
        lambda x: str(x) if isinstance(x, (np.datetime64, pd.Timestamp, pd.Timedelta)) else x
    ).to_dict()
    with open('experiment_results_fine_tuned.json', 'w') as f:
        json.dump(trials_df, f, indent=4)
