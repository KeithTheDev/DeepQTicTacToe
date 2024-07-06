import argparse
import random
import sys

import pygame
from tic_tac_toe.environment import TicTacToe
from tic_tac_toe.model import DQN
from tic_tac_toe.train import (choose_action, load_checkpoint, run_experiments,
                               train_dqn)
from tic_tac_toe_gameplay import run_pygame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play Tic-Tac-Toe with DQN")
    parser.add_argument('--mode', choices=['train', 'play', 'experiment'], required=True, help="Mode to run: train, play, or experiment")

    args = parser.parse_args()

    env = TicTacToe()
    architecture = 'default'
    policy_net = DQN(architecture=architecture)
    target_net = DQN(architecture=architecture)
    target_net.load_state_dict(policy_net.state_dict())

    if args.mode == 'train':
        best_hyperparameters = {
            'num_episodes': 10000000,
            #'num_episodes': 4923,
            'batch_size': 109,
            'gamma': 0.9007905885073516,
            'epsilon_start': 0.8982950447165381,
            'epsilon_end': 0.07793103251263457,
            'epsilon_decay': 416,
            'target_update': 24,
            'save_interval': 2669
        }

        train_dqn(
            env, policy_net, target_net,
            num_episodes=best_hyperparameters['num_episodes'],
            batch_size=best_hyperparameters['batch_size'],
            gamma=best_hyperparameters['gamma'],
            epsilon_start=best_hyperparameters['epsilon_start'],
            epsilon_end=best_hyperparameters['epsilon_end'],
            epsilon_decay=best_hyperparameters['epsilon_decay'],
            target_update=best_hyperparameters['target_update'],
            save_interval=best_hyperparameters['save_interval'],
            architecture=architecture
        )
    elif args.mode == 'experiment':
        run_experiments(num_experiments=10, num_episodes=5000)
    elif args.mode == 'play':
        load_checkpoint(policy_net, target_net, architecture=architecture)
        run_pygame(policy_net, target_net)
