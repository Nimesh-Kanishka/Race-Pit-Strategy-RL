# Race Pit Strategy Simulation for Reinforcement Learning

A custom Gymnasium environment that simulates a car race (inspired by NASCAR oval races). The goal is to finish the race as soon as possible. The RL agent has to balance going fast and conserving fuel/tires to achieve the best time and beat the other agents (cars).

# Versions

[v3](https://github.com/Nimesh-Kanishka/Race-Pit-Strategy-RL/tree/main/Race-Pit-Strategy-v3)

[v2](https://github.com/Nimesh-Kanishka/Race-Pit-Strategy-RL/tree/main/Race-Pit-Strategy-v2): Change environment to have multiple agents (cars) racing each other at the same time (using [Pettingzoo](https://pettingzoo.farama.org/content/basic_usage/) module)

[v1](https://github.com/Nimesh-Kanishka/Race-Pit-Strategy-RL/tree/main/Race-Pit-Strategy-v1): Change observation space from Dict to Box; Expand action space to include actions for determining the amount (percentage) to refuel and whether or not to change tires (during pitstops); Update maximum speed dynamically based on the condition of the tires

[v0](https://github.com/Nimesh-Kanishka/Race-Pit-Strategy-RL/tree/main/Race-Pit-Strategy-v0): Initial version
