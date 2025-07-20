import numpy as np
from .pit_strategy_agent import PitStrategyAgent

class CustomPitAgentv1(PitStrategyAgent):
    """
    Strategy:
    - Full throttle until fuel or tires drop below threshold.
    - Plan pit stops near end of a lap when below thresholds.
    - Refuel to full and change tires at every stop.
    """

    def __init__(
        self,
        fuel_threshold: float = 0.15,
        tire_threshold: float = 0.3
    ):
        super().__init__()

        # We will always use full throttle
        self.throttle = 1.0
        # Fraction of fuel left to trigger pit
        self.fuel_threshold = fuel_threshold
        # Fraction of tire life to trigger pit
        self.tire_threshold = tire_threshold

    def get_action(
        self,
        observation: np.ndarray
    ) -> np.ndarray:
        super().get_action(observation)

        # Determine if we need to pit next lap
        need_fuel = self.fuel <= self.fuel_threshold
        need_tires = self.tires <= self.tire_threshold
        # If we need service and are near lap end
        if (need_fuel or need_tires) and self.position >= 0.95:
            # Initiate pit entry
            self.pit_call = 1.0
            self.refuel_amount = 1.0 - self.fuel
            self.change_tires = 1.0
            return self._create_action_array()
        
        # Default actions        
        self.pit_call = 0.0
        self.refuel_amount = 0.0
        self.change_tires = 0.0
        return self._create_action_array()