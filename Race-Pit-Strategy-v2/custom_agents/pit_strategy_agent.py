from abc import ABC, abstractmethod
import numpy as np

class PitStrategyAgent(ABC):
    def __init__(self):
        # Observations
        self.lap = None
        self.position = None
        self.place = None
        self.distance_in_front = None
        self.distance_behind = None
        self.speed = None
        self.fuel = None
        self.tires = None
        self.in_pit = None
        self.pit_timer = None

        # Actions
        self.throttle = None
        self.pit_call = None
        self.refuel_amount = None
        self.change_tires = None

    @abstractmethod
    def get_action(self, observation: np.ndarray):
        self.lap = observation[0]
        self.position = observation[1]
        self.place = observation[2]
        self.distance_in_front = observation[3]
        self.distance_behind = observation[4]
        self.speed = observation[5]
        self.fuel = observation[6]
        self.tires = observation[7]
        self.in_pit = observation[8]
        self.pit_timer = observation[9]

    def _create_action_array(self):
        return np.array(
            [
                self.throttle,
                self.pit_call,
                self.refuel_amount,
                self.change_tires
            ],
            dtype=np.float32
        )
