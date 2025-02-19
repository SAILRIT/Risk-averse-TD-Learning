import numpy as np
import random
import matplotlib.pyplot as plt
import ot  # Add this for Sinkhorn transport calculations
import argparse
import warnings

warnings.filterwarnings("ignore")


# Parameters
alpha = 0.15  # Learning rate
lambda_ = 0.9  # Eligibility trace decay rate
epsilon = 0.7  # Initial epsilon
gamma = 0.99  # Discount factor
obstacle_fraction = 0.2
max_steps = 150  # Maximum steps per episode
num_episodes = 3000
epsilon_min = 0.00001  # Minimum epsilon

epsilon_decay_rate = 0.9  # Epsilon decays by this factor each episode

# Grid dimensions
M, N = 10, 10  # 10x10 grid
fuel_cost = 2
obstacle_cost = 10
goal_cost = 10


# Actions
ACTIONS = ["E", "W", "N", "S", "NE", "NW", "SE", "SW"]
NUM_ACTIONS = len(ACTIONS)



# -------------------
# Grid Environment
# -------------------
# def generate_grid(m, n, obstacle_fraction):
#     grid = np.zeros((m, n))
#     num_obstacles = int(obstacle_fraction * m * n)
#     obstacles = random.sample(range(m * n), num_obstacles)
#     for obs in obstacles:
#         x, y = divmod(obs, n)
#         grid[x, y] = 1
#     return grid
def generate_grid(m, n, obstacle_fraction, start_state, goal_state):

    grid = np.zeros((m, n))
    num_obstacles = int(obstacle_fraction * m * n)
    obstacle_positions = set()

    while len(obstacle_positions) < num_obstacles:
        # Randomly choose a cell
        x, y = random.randint(0, m - 1), random.randint(0, n - 1)

        # Ensure the cell is not the start state or goal state
        if (x, y) != start_state and (x, y) != goal_state:
            obstacle_positions.add((x, y))

    # Mark obstacles in the grid
    for x, y in obstacle_positions:
        grid[x, y] = 1

    return grid


def create_observation_space(m, n):
    """
    Observation space is the same as the state space but with probabilistic observations.
    """
    return [(x, y) for x in range(m) for y in range(n)]


def transition_model(state, action, grid):
    """
    Transition probabilities for state movement.
    """
    x, y = state
    m, n = grid.shape
    intended_move = {
        "E": (x, y + 1),
        "W": (x, y - 1),
        "N": (x - 1, y),
        "S": (x + 1, y),
        "NE": (x - 1, y + 1),
        "NW": (x - 1, y - 1),
        "SE": (x + 1, y + 1),
        "SW": (x + 1, y - 1)
    }
    adjacent_moves = {
        "E": [("NE", 0.05), ("SE", 0.05)],
        "W": [("NW", 0.05), ("SW", 0.05)],
        "N": [("NE", 0.05), ("NW", 0.05)],
        "S": [("SE", 0.05), ("SW", 0.05)],
        "NE": [("N", 0.05), ("E", 0.05)],
        "NW": [("N", 0.05), ("W", 0.05)],
        "SE": [("S", 0.05), ("E", 0.05)],
        "SW": [("S", 0.05), ("W", 0.05)],
    }
    transitions = []
    if action in intended_move:
        next_state = intended_move[action]
        if 0 <= next_state[0] < m and 0 <= next_state[1] < n:
            transitions.append((next_state, 0.9))
        else:
            transitions.append((state, 0.9))

    for adj_action, prob in adjacent_moves.get(action, []):
        adj_state = intended_move.get(adj_action, state)
        if 0 <= adj_state[0] < m and 0 <= adj_state[1] < n:
            transitions.append((adj_state, prob))
        else:
            transitions.append((state, prob))

    return transitions


def reward_model(state, goal, grid):
    x, y = state
    if state == goal:
        return goal_cost
    elif grid[x, y] == 1:
        return -obstacle_cost
    else:
        return -fuel_cost


def observation_model(state, grid):
    """
    Observation probabilities when the rover is adjacent to an obstacle.
    """
    x, y = state
    m, n = grid.shape
    observations = {}

    # Check if the rover is adjacent to any obstacle
    adjacent_cells = [
        (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),  # Orthogonal neighbors
        (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)  # Diagonal neighbors
    ]
    adjacent_obstacles = [cell for cell in adjacent_cells if 0 <= cell[0] < m and 0 <= cell[1] < n and grid[cell] == 1]

    if adjacent_obstacles:
        # The rover is adjacent to at least one obstacle
        for obs_cell in adjacent_obstacles:
            observations[obs_cell] = 0.6  # High probability of identifying the actual obstacle

        # Distribute the remaining probability (0.4) over all surrounding cells
        for cell in adjacent_cells + [state]:  # Include the current cell
            if cell not in observations:
                if 0 <= cell[0] < m and 0 <= cell[1] < n:
                    observations[cell] = 0.05  # Remaining cells get equal distribution
    else:
        # If not adjacent to an obstacle, return uniform observation
        for i in range(m):
            for j in range(n):
                observations[(i, j)] = 1 / (m * n)

    return observations


# -------------------
# Safety Term C(s, a)
# -------------------
def calculate_C(Q, T, observation, lambda_=1.0):
    epsilon = 1e-8

    q_values = [max(0, Q[observation][a]) for a in ACTIONS]
    t_values = [max(0, T.get((observation, a), 0)) for a in ACTIONS]

    q_sum = max(sum(q_values), epsilon)
    t_sum = max(sum(t_values), epsilon)
    q_values_norm = [q / q_sum for q in q_values]
    t_values_norm = [t / t_sum for t in t_values]

    reg = 0.1
    cost_matrix = (np.ones((NUM_ACTIONS, NUM_ACTIONS)) - np.eye(NUM_ACTIONS)) / (NUM_ACTIONS - 1)

    transport_plan = ot.sinkhorn(q_values_norm, t_values_norm, cost_matrix, reg)
    wasserstein_dist = max(np.sum(transport_plan * cost_matrix), epsilon)

    if wasserstein_dist == 0:
        return [0] * NUM_ACTIONS
    else:
        redistribution_amount = np.sum(transport_plan * (1 - np.eye(NUM_ACTIONS)), axis=1)
        received_amount = np.sum(transport_plan * (1 - np.eye(NUM_ACTIONS)), axis=0)
        abs_diff = np.abs(redistribution_amount - received_amount)

        return [(lambda_ * diff) / wasserstein_dist for diff in abs_diff]


def epsilon_greedy_with_C(Q, T, observation, epsilon):
    C_values = calculate_C(Q, T, observation)
    modified_q_values = [Q[observation][a] - 70* C_values[i] for i, a in enumerate(ACTIONS)]

    if random.uniform(0, 1) < epsilon:
        return random.choice(ACTIONS)
    else:
        best_idx = np.argmax(modified_q_values)
        return ACTIONS[best_idx]


# -------------------
# SARSA(λ) Algorithm
# -------------------
def initialize_q_t(obs_space):
    Q = {obs: {a: 0 for a in ACTIONS} for obs in obs_space}
    T = {}
    return Q, T


def sarsa_lambda(grid, start_state, goal_state, num_episodes, max_steps):
    obs_space = create_observation_space(M, N)
    Q, T = initialize_q_t(obs_space)  # Initialize Q-values and auxiliary T-values
    learning_curve = []
    num_hits_obstacles = 0
    num_reaches_goal = 0
    global epsilon

    for episode in range(num_episodes):
        # Initialize eligibility traces
        E = {obs: {a: 0 for a in ACTIONS} for obs in obs_space}

        state = start_state
        observations = observation_model(state, grid)
        observation = random.choices(list(observations.keys()), weights=list(observations.values()))[0]
        action = epsilon_greedy_with_C(Q, T, observation, epsilon)
        epsilon = max(epsilon_min, epsilon * epsilon_decay_rate)  # Decay epsilon
        total_reward = 0
        steps = 0

        while state != goal_state and steps < max_steps:
            transitions = transition_model(state, action, grid)
            next_state, prob = random.choices(transitions, weights=[p for _, p in transitions])[0]
            reward = reward_model(next_state, goal_state, grid)

            if grid[next_state[0], next_state[1]] == 1:
                num_hits_obstacles += 1
            if next_state == goal_state:
                num_reaches_goal += 1

            total_reward += reward

            # Get the next observation
            next_observations = observation_model(next_state, grid)
            next_observation = random.choices(list(next_observations.keys()), weights=list(next_observations.values()))[0]

            # Choose next action
            next_action = epsilon_greedy_with_C(Q, T, next_observation, epsilon)

            # SARSA(λ) Update
            td_error = reward + gamma * Q[next_observation][next_action] - Q[observation][action]

            # Update eligibility traces
            E[observation][action] += 1  # Increment trace for the current state-action pair

            # Update Q-values for all state-action pairs
            for obs in obs_space:
                for act in ACTIONS:
                    Q[obs][act] += alpha * td_error * E[obs][act]
                    E[obs][act] *= gamma * lambda_  # Decay eligibility traces

            # Update T
            T[(observation, action)] = reward + gamma * Q[next_observation][next_action]

            # Update state, observation, and action
            state = next_state
            observation = next_observation
            action = next_action
            steps += 1

        learning_curve.append(total_reward)
    print("Learning curve (returns for all episodes):", learning_curve)

    return Q, learning_curve, num_hits_obstacles, num_reaches_goal



parser = argparse.ArgumentParser(description="Takes an integer as random seed and runs the code")
parser.add_argument("-r", metavar="N", type=int, help="Index to pick from the rand_num")
args = parser.parse_args()

rand_num = list(range(1, 1000))
print("Number of elements in the random seed list %d" % len(rand_num))
print("The index from random seed list : %d" % args.r)
r = rand_num[args.r] if args.r is not None else 1
print("Value picked: %d" % r)

rand_num2 = [r]
start_state = (0, 0)
goal_state = (M - 1, N - 1)
grid = generate_grid(M, N, obstacle_fraction, start_state, goal_state)
# Variables to store cumulative results
all_learning_curves = []
all_hits_obstacles = []
all_reaches_goal = []
for r in rand_num2:
    # for seed in range(num_seeds):
        # print(f"Running simulation with random seed: {seed}")
        random.seed(r)
        np.random.seed(r)

        # Run SARSA(λ) for the current seed
        q_values, learning_curve, num_hits_obstacles, num_reaches_goal = sarsa_lambda(
            grid, start_state, goal_state, num_episodes, max_steps
        )

        # Store results
        all_learning_curves.append(learning_curve)
        all_hits_obstacles.append(num_hits_obstacles)
        all_reaches_goal.append(num_reaches_goal)

    # Compute averages
# avg_learning_curve = np.mean(all_learning_curves, axis=0)
avg_hits_obstacles = np.mean(all_hits_obstacles)
avg_reaches_goal = np.mean(all_reaches_goal)

# Print averaged results

print(f"Average number of hitting obstacles: {avg_hits_obstacles}")
print(f"Average number of reaching the goal: {avg_reaches_goal}")
