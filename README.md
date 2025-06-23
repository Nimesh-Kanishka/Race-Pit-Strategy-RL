# Race-Pit-Strategy-RL

A custom Gymnasium environment that simulates a car race. The goal is to finish the race as soon as possible. The RL agent has to balance going fast and conserving fuel/tires to achieve the best time possible.

v0: Initial version

v1: Change observation space from Dict to Box, Expand action space to include actions for determining the amount (percentage) to refuel and whether or not to change tires (during pitstops), Update maximum speed dynamically based on the condition of the tires
