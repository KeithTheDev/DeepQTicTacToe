# Tic-Tac-Toe DQN

## Overview

This project implements a Deep Q-Network (DQN) to play the game Tic-Tac-Toe using various neural network architectures. The project includes training the DQN agent, experimenting with hyperparameter tuning using Optuna, and playing the game using a trained model.

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/KeithTheDev/DeepQTicTacToe.git
    cd tic_tac_toe_dqn
    ```

2. **Create a virtual environment and activate it:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the DQN model:

```sh
python main.py --mode train
```

Running Experiments with Optuna

To run hyperparameter tuning experiments with Optuna:

1) Initialize the Optuna database:

```sh
python initialize_optuna_db.py
```

2) Run experiments:

```sh
python main.py --mode experiment
```
3) Use Optuna dashboard (optional but recommended for monitoring):

```sh
optuna-dashboard sqlite:///db.sqlite3
```
Open the provided URL in a web browser to monitor the optimization progress.

Playing the Game
To play the game against the trained DQN agent:

```sh
python main.py --mode play
```

## Project Structure

- `tic_tac_toe/`
  - `__init__.py`: Initializes the package.
  - `model.py`: Defines the DQN model with different architecture options.
  - `train.py`: Contains the training logic, optimization, and checkpoint management.
  - `environment.py`: Defines the Tic-Tac-Toe game environment.
  - `replay_buffer.py`: Implements a prioritized replay buffer for experience replay.
  
- `utils/`
  - `replay_buffer.py`: Alternative replay buffer implementation (if needed).

- `tic_tac_toe_gameplay.py`: Pygame interface to play Tic-Tac-Toe with the trained DQN agent.
- `initialize_optuna_db.py`: Script to initialize the Optuna database schema.
- `main.py`: Entry point to train the model, run experiments, or play the game.


## Explanation of Key Components

### DQN (Deep Q-Network)

The DQN model learns to play Tic-Tac-Toe by approximating the Q-value function, which estimates the expected future rewards for a given state-action pair. The model uses different neural network architectures to find the optimal policy.

### TD Lambda (Temporal Difference Lambda)

TD(λ) is a reinforcement learning method that blends TD learning and Monte Carlo methods. The λ parameter controls the trade-off between bias and variance:

- λ = 0: Equivalent to one-step TD learning (high bias, low variance).
- λ = 1: Equivalent to Monte Carlo methods (low bias, high variance).
- 0 < λ < 1: Mixes both approaches to balance bias and variance.

### Hyperparameters and Hyperparameter Tuning

Hyperparameters significantly influence the performance of machine learning models. Key hyperparameters in this project include the learning rate, batch size, discount factor (gamma), epsilon values for the epsilon-greedy policy, and the λ parameter in TD(λ).

#### Hyperparameter Tuning with Optuna:

Optuna is used for hyperparameter optimization to find the best hyperparameters for training the DQN model. The run_experiments function conducts multiple trials, each with different hyperparameter settings, to minimize the loss function. The optuna-dashboard helps monitor and analyze these experiments.

## Neural Network Architectures Summary

### Default Architecture
- **Layers**: 6 fully connected layers, each with 128 neurons.
- **Streams**: 
  - Value Stream: 1 output neuron.
  - Advantage Stream: 9 output neurons.

### Small Architecture
- **Layers**: 6 fully connected layers, each with 64 neurons.
- **Streams**: 
  - Value Stream: 1 output neuron.
  - Advantage Stream: 9 output neurons.

### Large Architecture
- **Layers**: 6 fully connected layers, each with 256 neurons.
- **Streams**: 
  - Value Stream: 1 output neuron.
  - Advantage Stream: 9 output neurons.

### Extra Small Architecture
- **Layers**: 6 fully connected layers, each with 32 neurons.
- **Streams**: 
  - Value Stream: 1 output neuron.
  - Advantage Stream: 9 output neurons.

### Medium Architecture
- **Layers**: 8 fully connected layers, each with 128 neurons.
- **Streams**: 
  - Value Stream: 1 output neuron.
  - Advantage Stream: 9 output neurons.

### Extra Large Architecture
- **Layers**: 6 fully connected layers, each with 512 neurons.
- **Streams**: 
  - Value Stream: 1 output neuron.
  - Advantage Stream: 9 output neurons.

### Deep Architecture
- **Layers**: 10 fully connected layers, each with 128 neurons.
- **Streams**: 
  - Value Stream: 1 output neuron.
  - Advantage Stream: 9 output neurons.

## Conclusion

This project provides a comprehensive implementation of a DQN agent for Tic-Tac-Toe, with robust training, evaluation, and experimentation capabilities. By exploring different architectures and tuning hyperparameters, it aims to develop an optimal agent for the game.