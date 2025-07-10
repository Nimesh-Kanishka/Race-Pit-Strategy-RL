import numpy as np
from .pit_strategy_agent import PitStrategyAgent

class RandomAgent(PitStrategyAgent):
    def get_action(
        self,
        observation: np.ndarray
    ) -> np.ndarray:
        super().get_action(observation)

        self.throttle = np.random.random()
        self.pit_call = np.random.random()
        self.refuel_amount = np.random.random()
        self.change_tires = np.random.random()

        return self._create_action_array()