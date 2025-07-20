import numpy as np
from .pit_strategy_agent import PitStrategyAgent

class CustomPitAgentv2(PitStrategyAgent):
    """
    Strategy:
    - Always run at full throttle for maximum on-track pace.
    - If the race goes under caution, take a "free" pit stop.
    - Under green-flag conditions, schedule pit stops only if fuel or tires drop
    below thresholds. Refuel just enough to finish the race. Always change tires.
    """

    def __init__(
        self,
        fuel_threshold: float = 0.15,
        tire_threshold: float = 0.3
    ):
        super().__init__()

        # We will always use full throttle
        self.throttle = 1.0
        # Fraction of fuel left to trigger pit under green flag conditions
        self.fuel_threshold = fuel_threshold
        # Fraction of tire life to trigger pit under green flag conditions
        self.tire_threshold = tire_threshold
        # Whether or not we pit during the current caution period
        self.this_caution_pit = False
        # Percentage of the race corresponding to one lap
        self.one_lap_percentage = None
        # Amount of fuel needed to run a lap at full throttle
        self.fuel_per_lap = None

    def get_action(
        self,
        observation: np.ndarray
    ) -> np.ndarray:
        super().get_action(observation)
        
        self.under_caution = bool(self.under_caution)
        self.in_pit = bool(self.in_pit)

        # Default actions        
        self.pit_call = 0.0
        self.refuel_amount = 0.0
        self.change_tires = 0.0

        # Calculate the fuel needed for a lap at full throttle
        if self.lap > 0 and self.one_lap_percentage is None:
            self.one_lap_percentage = self.lap
            self.fuel_per_lap = 1 - self.fuel

        # If we are under caution, we will pit even if we do not
        # need service as we can get a free pitstop
        if self.under_caution and not self.this_caution_pit:
            # If we have already pitted, set this_caution_pit to True
            # as we do not want to pit again in the next lap
            if self.in_pit:
                self.this_caution_pit = True
                return self._create_action_array()
            
            # Otherwise, we will make pit call
            self.pit_call = 1.0
            laps_left = (1 - self.lap) / self.one_lap_percentage
            fuel_needed = self.fuel_per_lap * laps_left
            self.refuel_amount = min(1.0 - self.fuel, fuel_needed)
            self.change_tires = 1.0
            return self._create_action_array()
        
        # Reset this_caution_pit when the caution ends
        elif not self.under_caution and self.this_caution_pit:
            self.this_caution_pit = False

        # Make pit call if we need service and are near lap end
        if self.position >= 0.95 and (
            self.fuel <= self.fuel_threshold or
            self.tires <= self.tire_threshold
        ):
            self.pit_call = 1.0
            laps_left = (1 - self.lap) / self.one_lap_percentage
            fuel_needed = self.fuel_per_lap * laps_left
            self.refuel_amount = min(1.0 - self.fuel, fuel_needed)
            self.change_tires = 1.0
            return self._create_action_array()
        
        return self._create_action_array()