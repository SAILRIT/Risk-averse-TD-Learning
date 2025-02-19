import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings("ignore")

#parameters

alpha = 0.15  # Learning rate
lambda_ = 0.9  # Eligibility trace decay rate
epsilon = 0.7  # Initial epsilon
epsilon_min = 0.00001

epsilon_decay_rate = 0.9  # Epsilon decays by this factor each episode
gamma = 0.99  # Discount factor

fuel_cost = 2
obstacle_cost = 10
goal_cost = 10
max_steps = 150  # Maximum steps per episode
num_episodes = 3000


# Grid dimensions
M, N = 10, 10  # 10x10 grid
obstacle_fraction = 0.2

# Actions
ACTIONS = ["E", "W", "N", "S", "NE", "NW", "SE", "SW"]




# def generate_grid(m, n, obstacle_fraction):
#     grid = np.zeros((m, n))
#     num_obstacles = int(obstacle_fraction * m * n)
#     obstacles = random.sample(range(m * n), num_obstacles)
#     for obs in obstacles:
#         x, y = divmod(obs, n)
#         grid[x, y] = 1
#     return grid
def generate_grid(m, n, obstacle_fraction, start_state, goal_state):
    """
    Generate a grid-world with obstacles, ensuring that the start state
    and goal state are not obstacles.

    Args:
    - m, n: Grid dimensions.
    - obstacle_fraction: Fraction of cells that should be obstacles.
    - start_state: (x, y) tuple for the start position.
    - goal_state: (x, y) tuple for the goal position.

    Returns:
    - A grid with obstacles.
    """
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


def choose_action(observation, q_values, epsilon):
    """
    Choose an action based on the observation and Q-values using epsilon-greedy policy.
    """
    if random.uniform(0, 1) < epsilon:
        return random.choice(ACTIONS)
    else:
        return max(ACTIONS, key=lambda action: q_values[observation][action])


def initialize_q_values(obs_space):
    """
    Initialize Q-values for each observation.
    """
    return {observation: {action: 0 for action in ACTIONS} for observation in obs_space}


def sarsa_lambda(grid, start_state, goal_state, num_episodes, max_steps):
    obs_space = create_observation_space(M, N)
    q_values = initialize_q_values(obs_space)
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
        action = choose_action(observation, q_values, epsilon)
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
            next_action = choose_action(next_observation, q_values, epsilon)

            # SARSA(λ) Update
            td_error = reward + gamma * q_values[next_observation][next_action] - q_values[observation][action]

            # Update eligibility traces
            E[observation][action] += 1  # Increment trace for the current state-action pair

            # Update Q-values for all state-action pairs
            for obs in obs_space:
                for act in ACTIONS:
                    q_values[obs][act] += alpha * td_error * E[obs][act]
                    E[obs][act] *= gamma * lambda_  # Decay eligibility traces

            # Update state, observation, and action
            state = next_state
            observation = next_observation
            action = next_action
            steps += 1

        learning_curve.append(total_reward)
    print("Learning curve (returns for all episodes):", learning_curve)

    return q_values, learning_curve, num_hits_obstacles, num_reaches_goal




parser = argparse.ArgumentParser(description="Takes an integer as random seed and runs the code")
parser.add_argument("-r", metavar="N", type=int, help="Index to pick from the rand_num")
args = parser.parse_args()

# # # Setting the random seed for reproducibility
rand_num = list(range(1, 1000))
print("Number of elements in the random seed list %d" % len(rand_num))
print("The index from random seed list : %d" % args.r)
r = rand_num[args.r] if args.r is not None else 1
print("Value picked: %d" % r)

rand_num2 = [r]

# Variables to store cumulative results across seeds
all_learning_curves = []
all_hits_obstacles = []
all_reaches_goal = []
for r in rand_num2:
    random.seed(r)
    np.random.seed(r)
    start_state = (0, 0)
    goal_state = (M - 1, N - 1)
    grid = generate_grid(M, N, obstacle_fraction, start_state, goal_state)
    # Run SARSA(λ) for the current seed
    q_values, learning_curve, num_hits_obstacles, num_reaches_goal = sarsa_lambda(
        grid, start_state, goal_state, num_episodes, max_steps
    )

    # Store results
    all_learning_curves.append(learning_curve)
    all_hits_obstacles.append(num_hits_obstacles)
    all_reaches_goal.append(num_reaches_goal)

    # Compute averages across seeds
    # avg_learning_curve = np.mean(all_learning_curves, axis=0)
    avg_hits_obstacles = np.mean(all_hits_obstacles)
    avg_reaches_goal = np.mean(all_reaches_goal)

    # Print averaged results
    # print(f"\nAveraged over {num_seeds} seeds:")
    print(f"Average number of hitting obstacles: {avg_hits_obstacles}")
    print(f"Average number of reaching the goal: {avg_reaches_goal}")


