from __future__ import annotations
import numpy as np
import random
import argparse
import ot
TRIALS = 1
# Environment Setup
GRID_WIDTH, GRID_HEIGHT = 7, 10
GOAL_POSITION = (0, 9)
START_POSITION = (0, 0)
CLIFF_REGION = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),
                    (1,1),(1,2),(1,3),(1,6),(1,7),(1,8),
                    (2,1),(2,2),(2,3),(2,6),(2,7),(2,8),
                    (3,1),(3,2),(3,3),(3,6),(3,7),(3,8)]


TRAP_REGION = [(1,4),(1,5),(2,4),(2,5),(3,4),(3,5)]  # Blue Region in the figure

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # [Left, Right, Down, Up]
# Assign action indices
LEFT, RIGHT, DOWN, UP = 0, 1, 2, 3

# Parameters
ALPHA = 0.1
GAMMA = 1
MAX_STEPS = 100
EPISODES = 1500
LAMBDA = 0.1
TRIALS = 1
EPSILON_START =1  # Start with a high epsilon for exploration
EPSILON_MIN = 0.0001  # Minimum value for epsilon
EPSILON_DECAY_RATE = 0.999  # Decay rate for epsilon
EPSILON = EPSILON_START


# Functions for the environment, policy, and training
def reset_Q():
    return np.zeros((GRID_WIDTH, GRID_HEIGHT, len(ACTIONS)))

def reset_T():
    # Initialize T(s,a) to zeros
    return np.zeros((GRID_WIDTH, GRID_HEIGHT, len(ACTIONS)))

def get_next_state(state, action):
    x, y = state
    dx, dy = ACTIONS[action]
    nx, ny = x + dx, y + dy
    if nx < 0 or nx >= GRID_WIDTH or ny < 0 or ny >= GRID_HEIGHT:
        return state  # Collision with wall
    if (nx, ny) in CLIFF_REGION:
        return START_POSITION  # Fall into the cliff, restart from start
    if (nx, ny) in TRAP_REGION:
        return (nx, ny + 1) if ny + 1 < GRID_HEIGHT else (nx, ny)  # Trap forces downward
    return (nx, ny)

def get_reward(state, next_state):
    if next_state == GOAL_POSITION:
        return 101  # Goal reached
    elif next_state in CLIFF_REGION:
        return -49  # Fall into the cliff
    else:
        return -1  # Normal movement cost

def calculate_C(Q, T, state, lambda_=1.0):
    q_values = Q[state[0], state[1], :]
    t_values = T[state[0], state[1], :]

    # Normalize q_values and t_values
    q_values_sum = np.sum(q_values)
    t_values_sum = np.sum(t_values)
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
    modified_q_values = Q[state[0], state[1], :] - 0.1* C_values  # Adjust Q-values with C-values
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, len(ACTIONS) - 1)
    else:
        return np.argmax(modified_q_values)


def epsilon_greedy(Q, state):
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, len(ACTIONS) - 1)  # Random action
    else:
        return np.argmax(Q[state[0], state[1], :])  # Best action

# Function to print the policy based on the Q-table
def print_policy(Q):
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            best_action = np.argmax(Q[x, y, :])
            action_symbol = ["L", "R", "D", "U"][best_action]  # Left, Right, Down, Up
            print(f"({x}, {y}): {action_symbol}")


def train_sarsa(state_visits):
    global EPSILON  # Allow modifying the global epsilon value
    Q = reset_Q()
    returns = []
    failures = 0 
    for episode in range(EPISODES):
        while True:
            state = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
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
            if next_state in CLIFF_REGION:
                state_visits[next_state[0], next_state[1]] += 1  # Count cliff state visit
                failures += 1
                break  # Episode ends if agent falls into the cliff

            state, action = next_state, next_action
        returns.append(total_reward)
        
        # Decay epsilon
        EPSILON = max(EPSILON * EPSILON_DECAY_RATE, EPSILON_MIN)

    # Print the policy after training
    print("Policy for SARSA:")
    print_policy(Q)
    print(f"Total cliff failures SARSA: {failures}")
    return returns

def train_sarsa_with_C(state_visits):
    global EPSILON  # Allow modifying the global epsilon value
    Q = reset_Q()
    T = reset_T()  # Initialize T(s,a)
    returns = []
    failures = 0 
    for episode in range(EPISODES):
        if episode % 50 == 0:
            print(f"Running episode: {episode}")
        while True:
            state = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
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

            if next_state in CLIFF_REGION:
                state_visits[next_state[0], next_state[1]] += 1  # Count cliff state visit
                failures += 1
                break  # Episode ends if agent falls into the cliff

            state, action = next_state, next_action
        returns.append(total_reward)
        
        # Decay epsilon
        EPSILON = max(EPSILON * EPSILON_DECAY_RATE, EPSILON_MIN)

    # Print the policy after training
    print("Policy for Safe SARSA")
    print_policy(Q)
    print(f"Total cliff failures Safe SARSA: {failures}")
    return returns


def train_sarsa_lambda(state_visits):
    Q = reset_Q()
    returns = []
    failures = 0 
    for episode in range(EPISODES):
        global EPSILON
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY_RATE)  # Decay epsilon
        while True:
            state = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if state != GOAL_POSITION:
                break
        action = epsilon_greedy(Q, state)
        eligibility_trace = np.zeros((GRID_WIDTH, GRID_HEIGHT, len(ACTIONS)))
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

            for x in range(GRID_WIDTH):
                for y in range(GRID_HEIGHT):
                    for a in range(len(ACTIONS)):
                        Q[x, y, a] += ALPHA * delta * eligibility_trace[x, y, a]
                        eligibility_trace[x, y, a] *= GAMMA * LAMBDA

            if next_state == GOAL_POSITION:
                state_visits[next_state[0], next_state[1]] += 1  # Count goal state visit
                break  # Episode ends if goal is reached

            if next_state in CLIFF_REGION:
                state_visits[next_state[0], next_state[1]] += 1  # Count cliff state visit
                failures += 1
                break  # Episode ends if agent falls into the cliff


            state, action = next_state, next_action

        returns.append(total_reward)

    # Print the policy after training
    print("Policy for SARSA(lambda):")
    print_policy(Q)
    print(f"Total cliff failures SARSA(lambda): {failures}")
    return returns


def train_q_learning(state_visits):
    global EPSILON  # Allow modifying the global epsilon value
    Q = reset_Q()
    returns = []
    failures = 0 
    for episode in range(EPISODES):
        while True:
            state = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
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

            if next_state in CLIFF_REGION:
                state_visits[next_state[0], next_state[1]] += 1  # Count cliff state visit
                failures += 1
                break  # Episode ends if agent falls into the cliff

            state = next_state
        returns.append(total_reward)
        
        # Decay epsilon
        EPSILON = max(EPSILON * EPSILON_DECAY_RATE, EPSILON_MIN)

    # Print the policy after training
    print("Policy for Q-learning:")
    print_policy(Q)
    print(f"Total cliff failures Q-learning: {failures}")
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
    state_visits_sarsa = np.zeros((GRID_WIDTH, GRID_HEIGHT))

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
    state_visits_q_learning = np.zeros((GRID_WIDTH, GRID_HEIGHT))

    # Run Q-learning
    for trial in range(TRIALS):
        returns = train_q_learning(state_visits_q_learning)
        q_learning_returns[trial] = returns
        print(f"Trial {trial + 1} Q-learning Returns (Random Seed: {r}): {returns}")
    # Print mean state visit distribution for Q-learning
    print(f"Mean State Visits after {EPISODES} episodes (Q-learning):\n{state_visits_q_learning / EPISODES}")

# for r in rand_num2:
#     np.random.seed(r)
#     sarsa_L_returns = np.zeros((TRIALS, EPISODES))
#     state_L_visits_sarsa = np.zeros((GRID_WIDTH, GRID_HEIGHT))
#     for trial in range(TRIALS):
#         returns = train_sarsa_lambda(state_L_visits_sarsa)
#         sarsa_L_returns[trial] = returns
#         print(f"Trial {trial + 1} SARSA Lambda Returns (Random Seed: {r}): {returns}")
#     # Print mean state visit distribution for SARSA
#     print(f"Mean State Visits after {EPISODES} episodes (SARSA Lambda):\n{state_L_visits_sarsa / EPISODES}")


for r in rand_num2:
    np.random.seed(r)
    sarsa_with_C_returns = np.zeros((TRIALS, EPISODES))
    state_visits_sarsa_with_C = np.zeros((GRID_WIDTH, GRID_HEIGHT))
    for trial in range(TRIALS):
        returns = train_sarsa_with_C(state_visits_sarsa_with_C)
        sarsa_with_C_returns[trial] = returns
        print(f"Trial {trial + 1} SARSA with C Returns (Random Seed: {r}): {returns}")
    # Print mean state visit distribution for SARSA with C
    print(f"Mean State Visits after {EPISODES} episodes (SARSA with C):\n{state_visits_sarsa_with_C / EPISODES}")
