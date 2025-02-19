from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib as mpl
# import logging
import numpy as np
import gymnasium as gym
# from gymnasium import spaces
import sys
import ot
from scipy.linalg import eig
import warnings
import time
import argparse
import pandas as pd
warnings.filterwarnings("ignore")
import random
from tqdm import tqdm  # For progress bar during trials
import os

# Grid dimensions and environment setup
GRID_SIZE = 10
GOAL_POSITION = (9, 9)

# slippery_states = (
#     [(i, j) for i in range(0, 3) for j in range(2, 5)] +
#     [(i, j) for i in range(4, 7) for j in range(1, 5)] 
# )

slippery_states = (
    [(2,0), (2,1),
    (3,0), (3,1),
    (4,0), (4,1),
    (5,0), (5,1),
    (6,0), (6,1),
    (7,0), (7,1),
    (8,0), (8,1),
    (4,4), (4,5),
    (5,4), (5,5),
    (1,8), (1,9),
    (2,8), (2,9),
    (3,8), (3,9),
    (6,7), (6,8),(6,9),
    (7,7),(7,8), (7,9),
    (8,7),(8,8), (8,9),
    ] 
)

slippery_reward_range = (-12, 10)
ACTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]

# SARSA parameters
ALPHA = 0.1
GAMMA = 1
# EPSILON = 0.1
MAX_STEPS = 100
EPISODES = 1500
TRIALS = 1
EPSILON_START =1  # Start with a high epsilon for exploration
EPSILON_MIN = 0.001  # Minimum value for epsilon
EPSILON_DECAY_RATE = 0.999  # Decay rate for epsilon
EPSILON = EPSILON_START
LAMBDA = 0.02
# high_cost_value=40

# Functions for the environment, policy, and training
def reset_Q():
    return np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

def reset_T():
    # Initialize T(s,a) to zeros
    return np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))


def get_next_state(state, action):
    x, y = state
    dx, dy = ACTIONS[action]
    nx, ny = x + dx, y + dy
    if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE:
        return state  # Collision with wall
    return (nx, ny)

def get_reward(state, next_state):
    if next_state == GOAL_POSITION:
        return -1  # Goal reached
    elif state == next_state:
        return -10  # Attempt to move out of the grid-world
    elif state in slippery_states:
        return random.uniform(*slippery_reward_range)  # Transition out of slippery state
    else:
        return -1  # Normal movement cost

# def calculate_C(Q, state, lambda_=1.0):
#     target_values = np.zeros(len(ACTIONS))
#     for action in range(len(ACTIONS)):
#         next_state = get_next_state(state, action)
#         reward = get_reward(state, next_state)
#         next_action = epsilon_greedy(Q, next_state)
#         target_values[action] = reward + GAMMA * Q[next_state[0], next_state[1], next_action]

#     q_values = Q[state[0], state[1], :]

def calculate_C(Q, T, state, lambda_=1.0):
    q_values = Q[state[0], state[1], :]
    t_values = T[state[0], state[1], :]

    # Normalize q_values and t_values
    q_values_sum = np.sum(q_values)
    t_values_sum = np.sum(t_values)

    # Avoid division by zero by adding a small epsilon if the sum is zero
    epsilon = 1e-8
    q_values_sum = q_values_sum if q_values_sum != 0 else epsilon
    t_values_sum = t_values_sum if t_values_sum != 0 else epsilon
    q_values_norm = q_values / q_values_sum
    t_values_norm = t_values / t_values_sum

    reg = 0.005  # Base regularization parameter
    reg_factor = max(np.std(q_values_norm), np.std(t_values_norm), epsilon)
    reg_adaptive = reg * reg_factor
    cost_matrix = np.ones((len(ACTIONS), len(ACTIONS))) - np.eye(len(ACTIONS))
    # Compute the transport plan using the Sinkhorn algorithm with adaptive regularization
    transport_plan = ot.sinkhorn(q_values_norm, t_values_norm, cost_matrix, reg=reg_adaptive)

    wasserstein_dist = np.sum(transport_plan * cost_matrix)

    if wasserstein_dist == 0:
        return np.zeros(len(ACTIONS))
    else:
        redistribution_amount = np.sum(transport_plan * (1 - np.eye(len(ACTIONS))), axis=1)
        received_amount = np.sum(transport_plan * (1 - np.eye(len(ACTIONS))), axis=0)
        abs_diff = np.abs(redistribution_amount - received_amount)
        C_values = (lambda_ * abs_diff) / wasserstein_dist

        return C_values



def epsilon_greedy_with_C(Q, T, state):
    C_values = calculate_C(Q, T, state)
    modified_q_values = Q[state[0], state[1], :] - 0.8* C_values  # Adjust Q-values with C-values
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, len(ACTIONS) - 1)
    else:
        return np.argmax(modified_q_values)


def epsilon_greedy(Q, state):
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, len(ACTIONS) - 1)  # Random action
    else:
        return np.argmax(Q[state[0], state[1], :])  # Best action

# ACTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]
# Function to print the policy based on the Q-table
def print_policy(Q):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            best_action = np.argmax(Q[x, y, :])
            action_symbol = ["R", "L", "D", "U"][best_action]  # Right, Left, Up, Down
            print(f"({x}, {y}): {action_symbol}")

def train_sarsa(state_visits):
    global EPSILON  # Allow modifying the global epsilon value
    Q = reset_Q()
    returns = []
    for episode in range(EPISODES):
        while True:
            state = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if state != GOAL_POSITION:
                break
        action = epsilon_greedy(Q, state)
        total_reward = 0
        for _ in range(MAX_STEPS):
            state_visits[state[0], state[1]] += 1  # Increment visit count for current state
            next_state = get_next_state(state, action)
            reward = get_reward(state, next_state)
            next_action = epsilon_greedy(Q, next_state)
            total_reward += reward
            # SARSA update
            Q[state[0], state[1], action] += ALPHA * (
                reward + GAMMA * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action]
            )
            if next_state == GOAL_POSITION:
                state_visits[next_state[0], next_state[1]] += 1  # Count goal state visit
                break  # Episode ends if goal is reached
            state, action = next_state, next_action
        returns.append(total_reward)
        
        # Decay epsilon
        EPSILON = max(EPSILON * EPSILON_DECAY_RATE, EPSILON_MIN)

    # Print the policy after training
    print("Policy for SARSA:")
    print_policy(Q)
    return returns

# Modify the train_sarsa_with_C function to include epsilon decay
def train_sarsa_with_C(state_visits):
    global EPSILON  # Allow modifying the global epsilon value
    Q = reset_Q()
    T = reset_T()  # Initialize T(s,a)
    returns = []
    for episode in range(EPISODES):
        if episode % 50 == 0:
            print(f"Running episode: {episode}")
        while True:
            state = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if state != GOAL_POSITION:
                break
        action = epsilon_greedy_with_C(Q, T, state)
        total_reward = 0
        for _ in range(MAX_STEPS):
            state_visits[state[0], state[1]] += 1  # Increment visit count for current state
            next_state = get_next_state(state, action)
            reward = get_reward(state, next_state)
            next_action = epsilon_greedy_with_C(Q, T, next_state)
            total_reward += reward

            # Update T(s,a)
            T[state[0], state[1], action] = reward + GAMMA * np.max(Q[next_state[0], next_state[1], :])

            # SARSA update
            Q[state[0], state[1], action] += ALPHA * (
                reward + GAMMA * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action]
            )
            if next_state == GOAL_POSITION:
                state_visits[next_state[0], next_state[1]] += 1  # Count goal state visit
                break  # Episode ends if goal is reached
            state, action = next_state, next_action
        returns.append(total_reward)
        
        # Decay epsilon
        EPSILON = max(EPSILON * EPSILON_DECAY_RATE, EPSILON_MIN)

    # Print the policy after training
    print("Policy for Safe SARSA")
    print_policy(Q)
    return returns


def train_expected_sarsa(state_visits):
    global EPSILON  # Allow modifying the global epsilon value
    Q = reset_Q()
    returns = []
    for episode in range(EPISODES):
        while True:
            state = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if state != GOAL_POSITION:
                break
        total_reward = 0
        for _ in range(MAX_STEPS):
            state_visits[state[0], state[1]] += 1  # Increment visit count for current state
            action = epsilon_greedy(Q, state)

            # Perform action and observe the next state and reward
            next_state = get_next_state(state, action)
            reward = get_reward(state, next_state)
            total_reward += reward

            # Compute expected Q-value for the next state
            expected_q_next = 0
            for a in range(len(ACTIONS)):
                if a == np.argmax(Q[next_state[0], next_state[1], :]):
                    prob = 1 - EPSILON + (EPSILON / len(ACTIONS))
                else:
                    prob = EPSILON / len(ACTIONS)
                expected_q_next += prob * Q[next_state[0], next_state[1], a]

            # Update Q-value using Expected SARSA
            Q[state[0], state[1], action] += ALPHA * (
                reward + GAMMA * expected_q_next - Q[state[0], state[1], action]
            )

            if next_state == GOAL_POSITION:
                state_visits[next_state[0], next_state[1]] += 1  # Count goal state visit
                break  # Episode ends if goal is reached

            state = next_state
        returns.append(total_reward)

        # Decay epsilon
        EPSILON = max(EPSILON * EPSILON_DECAY_RATE, EPSILON_MIN)

    # Print the policy after training
    print("Policy for Expected SARSA:")
    print_policy(Q)
    return returns


def train_sarsa_lambda(state_visits):
    Q = reset_Q()
    returns = []
    for episode in range(EPISODES):
        global EPSILON
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY_RATE)  # Decay epsilon
        state = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        while state == GOAL_POSITION:
            state = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        action = epsilon_greedy(Q, state)
        eligibility_trace = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
        total_reward = 0

        for _ in range(MAX_STEPS):
            state_visits[state[0], state[1]] += 1  # Increment visit count for current state
            next_state = get_next_state(state, action)
            reward = get_reward(state, next_state)
            next_action = epsilon_greedy(Q, next_state)
            total_reward += reward

            # SARSA(lambda) update
            delta = reward + GAMMA * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action]
            eligibility_trace[state[0], state[1], action] += 1

            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    for a in range(len(ACTIONS)):
                        Q[x, y, a] += ALPHA * delta * eligibility_trace[x, y, a]
                        eligibility_trace[x, y, a] *= GAMMA * LAMBDA

            if next_state == GOAL_POSITION:
                state_visits[next_state[0], next_state[1]] += 1  # Count goal state visit
                break  # Episode ends if goal is reached

            state, action = next_state, next_action

        returns.append(total_reward)

    # Print the policy after training
    print("Policy for SARSA(lambda):")
    print_policy(Q)
    return returns


def train_q_learning_with_C(state_visits):
    global EPSILON  # Allow modifying the global epsilon value
    Q = reset_Q()
    T = reset_T()  # Initialize T(s,a)
    returns = []
    for episode in range(EPISODES):
        if episode % 50 == 0:
            print(f"Running episode: {episode}")
        while True:
            state = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if state != GOAL_POSITION:
                break
        total_reward = 0
        for _ in range(MAX_STEPS):
            state_visits[state[0], state[1]] += 1  # Increment visit count for current state
            action = epsilon_greedy_with_C(Q, T, state)  # Use modified Q-values with C
            next_state = get_next_state(state, action)
            reward = get_reward(state, next_state)
            total_reward += reward

            # Update T(s,a)
            T[state[0], state[1], action] = reward + GAMMA * np.max(Q[next_state[0], next_state[1], :])

            # Compute C-values
            # C_values = calculate_C(Q, T, state)

            # Q-learning update with C-values
            Q[state[0], state[1], action] += ALPHA * (
                reward + GAMMA * np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action]
            )

            if next_state == GOAL_POSITION:
                state_visits[next_state[0], next_state[1]] += 1  # Count goal state visit
                break  # Episode ends if goal is reached
            state = next_state
        returns.append(total_reward)
        
        # Decay epsilon
        EPSILON = max(EPSILON * EPSILON_DECAY_RATE, EPSILON_MIN)

    # Print the policy after training
    print("Policy for Safe Q-learning:")
    print_policy(Q)
    return returns


def train_q_learning(state_visits):
    global EPSILON  # Allow modifying the global epsilon value
    Q = reset_Q()
    returns = []
    for episode in range(EPISODES):
        while True:
            state = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if state != GOAL_POSITION:
                break
        total_reward = 0
        for _ in range(MAX_STEPS):
            state_visits[state[0], state[1]] += 1  # Increment visit count for current state
            action = epsilon_greedy(Q, state)
            next_state = get_next_state(state, action)
            reward = get_reward(state, next_state)
            total_reward += reward
            # Q-learning update
            Q[state[0], state[1], action] += ALPHA * (
                reward + GAMMA * np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action]
            )
            if next_state == GOAL_POSITION:
                state_visits[next_state[0], next_state[1]] += 1  # Count goal state visit
                break  # Episode ends if goal is reached
            state = next_state
        returns.append(total_reward)
        
        # Decay epsilon
        EPSILON = max(EPSILON * EPSILON_DECAY_RATE, EPSILON_MIN)

    # Print the policy after training
    print("Policy for Q-learning:")
    print_policy(Q)
    return returns




# Set lambda to 0.3 for eligibility traces
LAMBDA1 = 0.3

def train_sarsa_lambda_with_C(state_visits):
    global EPSILON  # Allow modifying the global epsilon value
    Q = reset_Q()
    T = reset_T()  # Initialize T(s,a)
    returns = []
    for episode in range(EPISODES):
        if episode % 50 == 0:
            print(f"Running episode: {episode}")
        while True:
            state = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if state != GOAL_POSITION:
                break
        action = epsilon_greedy_with_C(Q, T, state)
        eligibility_trace = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
        total_reward = 0

        for _ in range(MAX_STEPS):
            state_visits[state[0], state[1]] += 1  # Increment visit count for current state
            next_state = get_next_state(state, action)
            reward = get_reward(state, next_state)
            next_action = epsilon_greedy_with_C(Q, T, next_state)
            total_reward += reward

            # Update T(s,a)
            T[state[0], state[1], action] = reward + GAMMA * np.max(Q[next_state[0], next_state[1], :])

            # Compute delta
            delta = reward + GAMMA * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action]
            eligibility_trace[state[0], state[1], action] += 1

            # Update Q-values using eligibility traces
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    for a in range(len(ACTIONS)):
                        Q[x, y, a] += ALPHA * delta * eligibility_trace[x, y, a]
                        eligibility_trace[x, y, a] *= GAMMA * LAMBDA1

            if next_state == GOAL_POSITION:
                state_visits[next_state[0], next_state[1]] += 1  # Count goal state visit
                break  # Episode ends if goal is reached
            state, action = next_state, next_action
        returns.append(total_reward)
        
        # Decay epsilon
        EPSILON = max(EPSILON * EPSILON_DECAY_RATE, EPSILON_MIN)

    # Print the policy after training
    print("Policy for Safe SARSA with Eligibility Traces:")
    print_policy(Q)
    return returns



##################### Run ######################

rand_num = list(range(1, 10000))

parser = argparse.ArgumentParser(description='Takes an integer as random seed and runs the code')
parser.add_argument('-r', metavar='N', type=int, help='Index to pick from the rand_num')

args = parser.parse_args()
print("Number of elements in the random seed list %d" % len(rand_num))
print("The index from random seed list : %d" % args.r)
print("Value picked: %d" % rand_num[args.r])

rand_num2 = [rand_num[args.r]]
for r in rand_num2:
    np.random.seed(r)
    sarsa_returns = np.zeros((TRIALS, EPISODES))
    state_visits_sarsa = np.zeros((GRID_SIZE, GRID_SIZE))

    # Run SARSA
    for trial in range(TRIALS):
        returns = train_sarsa(state_visits_sarsa)
        sarsa_returns[trial] = returns
        print(f"Trial {trial + 1} SARSA Returns (Random Seed: {r}): {returns}")
    # Print mean state visit distribution for SARSA
    print(f"Mean State Visits after {EPISODES} episodes (SARSA):\n{state_visits_sarsa / EPISODES}")

for r in rand_num2:
    np.random.seed(r)
    q_learning_returns = np.zeros((TRIALS, EPISODES))
    state_visits_q_learning = np.zeros((GRID_SIZE, GRID_SIZE))

    # Run Q-learning
    for trial in range(TRIALS):
        returns = train_q_learning(state_visits_q_learning)
        q_learning_returns[trial] = returns
        print(f"Trial {trial + 1} Q-learning Returns (Random Seed: {r}): {returns}")
    # Print mean state visit distribution for Q-learning
    print(f"Mean State Visits after {EPISODES} episodes (Q-learning):\n{state_visits_q_learning / EPISODES}")

for r in rand_num2:
    np.random.seed(r)
    sarsa_L_returns = np.zeros((TRIALS, EPISODES))
    state_L_visits_sarsa = np.zeros((GRID_SIZE, GRID_SIZE))
    for trial in range(TRIALS):
        returns = train_sarsa_lambda(state_L_visits_sarsa)
        sarsa_L_returns[trial] = returns
        print(f"Trial {trial + 1} SARSA Lambda Returns (Random Seed: {r}): {returns}")
    # Print mean state visit distribution for SARSA
    print(f"Mean State Visits after {EPISODES} episodes (SARSA Lambda):\n{state_L_visits_sarsa / EPISODES}")


for r in rand_num2:
    np.random.seed(r)
    expected_sarsa_returns = np.zeros((TRIALS, EPISODES))
    state_visits_expected_sarsa = np.zeros((GRID_SIZE, GRID_SIZE))

    # Run Expected SARSA
    for trial in range(TRIALS):
        returns = train_expected_sarsa(state_visits_expected_sarsa)
        expected_sarsa_returns[trial] = returns
        print(f"Trial {trial + 1} Expected SARSA Returns (Random Seed: {r}): {returns}")
    # Print mean state visit distribution for Expected SARSA
    print(f"Mean State Visits after {EPISODES} episodes (Expected SARSA):\n{state_visits_expected_sarsa / EPISODES}")


for r in rand_num2:
    np.random.seed(r)
    sarsa_with_C_returns = np.zeros((TRIALS, EPISODES))
    state_visits_sarsa_with_C = np.zeros((GRID_SIZE, GRID_SIZE))
    for trial in range(TRIALS):
        returns = train_sarsa_with_C(state_visits_sarsa_with_C)
        sarsa_with_C_returns[trial] = returns
        print(f"Trial {trial + 1} SARSA with C Returns (Random Seed: {r}): {returns}")
    # Print mean state visit distribution for SARSA with C
    print(f"Mean State Visits after {EPISODES} episodes (SARSA with C):\n{state_visits_sarsa_with_C / EPISODES}")


for r in rand_num2:
    np.random.seed(r)
    safe_q_learning_returns = np.zeros((TRIALS, EPISODES))
    state_visits_safe_q_learning = np.zeros((GRID_SIZE, GRID_SIZE))

    # Run Safe Q-learning
    for trial in range(TRIALS):
        returns = train_q_learning_with_C(state_visits_safe_q_learning)
        safe_q_learning_returns[trial] = returns
        print(f"Trial {trial + 1} Safe Q-learning Returns (Random Seed: {r}): {returns}")
    # Print mean state visit distribution for Safe Q-learning
    print(f"Mean State Visits after {EPISODES} episodes (Safe Q-learning):\n{state_visits_safe_q_learning / EPISODES}")



for r in rand_num2:
    np.random.seed(r)
    sarsa_lambda_with_C_returns = np.zeros((TRIALS, EPISODES))
    state_visits_sarsa_lambda_with_C = np.zeros((GRID_SIZE, GRID_SIZE))
    for trial in range(TRIALS):
        returns = train_sarsa_lambda_with_C(state_visits_sarsa_lambda_with_C)
        sarsa_lambda_with_C_returns[trial] = returns
        print(f"Trial {trial + 1} Safe SARSA with Eligibility Traces Returns (Random Seed: {r}): {returns}")
    # Print mean state visit distribution for Safe SARSA with Eligibility Traces
    print(f"Mean State Visits after {EPISODES} episodes (Safe SARSA with Eligibility Traces):\n{state_visits_sarsa_lambda_with_C / EPISODES}")
