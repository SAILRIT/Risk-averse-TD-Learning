# Overview
This is the code release for the paper Risk-Averse Reinforcement Learning: An Optimal Transport Perspective on TD Learning.  


## Case Studies

**Dependencies:** `numpy`, `pot`

### Grid-world with reward uncertainty
We consider a 10×10 grid-world environment with normal, goal, and slippery states. The agent can move up, down, left, and right. For any movement to a normal state, the agent receives the reward of −1, while transitions to slippery states result in a random reward in the range [−12, 10]. Collisions with walls incur a reward of −10. The episode terminates when the agent either reaches the goal state in the top-right corner or completes a maximum of 100 steps.

![image](https://github.com/user-attachments/assets/36a27626-8ee1-43ac-8277-221cfb675db9)


### Cliff walking with transition uncertainty
The cliff walking environment consists of three zones: the cliff region, the trap region, and the feasible region. The agent starts at the bottom-left corner to reach the goal at the bottom-right corner while avoiding the cliff zone, which represents unsafe states. Entering the cliff region results in task failure. The agent can move freely within the feasible region in four directions: up, down, left, and right. Entering the trap region forces the agent to move downward, regardless of its chosen action, eventually ending up in the cliff region. Each movement yields a reward of −1. If the agent collides with the environment borders, its position remains unchanged, but it still earns the movement reward. Reaching the target earns the agent a reward of 101, while entering the cliff region results in a −49 penalty.

![image](https://github.com/user-attachments/assets/fd1e708b-e631-4e52-813b-aca5a741a705)


### Rover navigation with partial observability
In this case study, a rover must navigate a two-dimensional terrain map represented as a 10×10 grid, where 3 of the grid cells are obstacles. Each grid cell represents a state, and the rover can move in eight geographic directions. However, the environment is stochastic; for example, as shown in Fig. 3, when the rover takes the action east, it moves to the intended grid cell with a probability of 0.9 but may move to one of the adjacent cells with a probability of 0.05. Partial observability exists because the rover cannot directly detect the locations of obstacle cells through its measurements. When the rover moves to a cell adjacent to an obstacle, it can identify the exact location of the obstacle (marked in magenta) with a probability of 0.6 and observe a probability distribution over nearby cells (marked in pink). Colliding with an obstacle results in an immediate penalty of 10, while reaching the goal region provides no immediate reward. All other grid cells impose a penalty of 2.


![image](https://github.com/user-attachments/assets/7b12cc75-529d-4306-bf4d-4a843c24af60)





