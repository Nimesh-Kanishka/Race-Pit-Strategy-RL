import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import pygame

# Register this module as a gym environment.
# When running this module, the warning "UserWarning: WARN: Overriding
# environment RacePitStrategy-v1 already in registry." can be ignored.
register(
    id="RacePitStrategy-v1",
    entry_point="race_pit_strategy:RacePitStrategyEnv", # module_name:class_name
    max_episode_steps=2500,
)

class RacePitStrategyEnv(gym.Env):
    metadata = {"render_modes": ["human", "terminal"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        num_laps: int = 50,
        track_length: float = 2500.0,
        max_speed: float = 75.0,
        fuel_capacity: float = 100.0,
        fuel_consumption_rate: float = 0.5,
        tire_wear_rate: float = 0.0015,
        base_pit_stop_time: int = 15,
        refuel_time_per_unit: float = 0.06,
        tire_change_time: int = 10
    ):
        super().__init__()

        self.render_mode = render_mode

        # Observation space
        # Consists of 7 observations: Current lap (Normalized), Position on track (Normalized),
        #                             Speed (Normalized), Fuel left (Normalized), Tire left,
        #                             Whether in pitlane or not, Pit timer (Normalized)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        # Action space
        # Consists of 4 actions: Throttle, Pit call, Refuel amount, Change tires
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Environment parameters
        self.num_laps = num_laps
        self.track_length = track_length
        self.max_speed = max_speed
        # We will penalize the agent if the speed is below a certain threshold. Otherwise the agent
        # may stop on the track and wait for the simulation to end to minimize other penalties.
        self.min_speed = max_speed / 1.15
        self.fuel_capacity = fuel_capacity
        self.fuel_consumption_rate = fuel_consumption_rate
        self.tire_wear_rate = tire_wear_rate
        self.base_pit_stop_time = base_pit_stop_time
        self.refuel_time_per_unit = refuel_time_per_unit
        self.tire_change_time = tire_change_time
        self.best_time = math.ceil(track_length / max_speed)

        # State variables
        self._init_state()

        # Pygame variables
        self.screen_size = None
        self.screen = None
        self.clock = None
        self.center = None
        self.track_radius = None

    def _init_state(self):
        self.lap = 0
        self.position = 0.0
        self.lap_time = 0
        self.speed = 0.0
        self.fuel = self.fuel_capacity
        self.tires = 1.0
        self.in_pit = False
        self.refuel_units = 0
        self.change_tires = False
        self.pit_time = 0
        self.pit_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset state variables
        self._init_state()

        self.render()

        return self._get_obs(), \
            self._get_info()

    def _get_obs(self):
        return np.array(
            [
                self.lap / self.num_laps,
                self.position / self.track_length,
                self.speed / self.max_speed,
                self.fuel / self.fuel_capacity,
                self.tires,
                float(self.in_pit),
                self.pit_timer / self.pit_time if self.in_pit else 0.0
            ],
            dtype=np.float32
        )
    
    def _get_info(self, finish=False):
        return {
            "finish": finish,
            "lap": self.lap,
            "position": self.position,
            "fuel": self.fuel,
            "tires": self.tires
        }

    def step(self, action):
        reward = 0.0
        terminated = False
        finish = False
        
        throttle = np.clip(action[0], 0.0, 1.0)
        pit_call = bool(np.round(np.clip(action[1], 0.0, 1.0)))
        refuel_amount = np.clip(action[2], 0.0, 1.0)
        change_tires = bool(np.round(np.clip(action[3], 0.0, 1.0)))

        self.lap_time += 1

        if self.in_pit:
            self.pit_timer += 1

            if self.pit_timer == self.pit_time:
                self.fuel = min(self.fuel + self.refuel_units, self.fuel_capacity)
                self.tires = 1.0 if self.change_tires else self.tires
                self.in_pit = False
                self.refuel_units = 0
                self.change_tires = False
                self.pit_time = 0
                self.pit_timer = 0

        else:
            # If we have run out of fuel or worn out tires
            if self.fuel == 0 or self.tires == 0:
                self.speed = 0.0

                # Penalty for running out of fuel or worn out tires
                reward -= 25.0

                terminated = True

            else:
                max_speed = self.max_speed * \
                    (((0.55 * self.tires - 1.75) * self.tires + 1.9) * self.tires + 0.3)

                self.speed = min(max_speed * throttle,
                                 self.fuel * max_speed / self.fuel_consumption_rate,
                                 self.tires * max_speed / self.tire_wear_rate)
                
                self.fuel = max(0.0, self.fuel - self.fuel_consumption_rate * throttle)
                self.tires = max(0.0, self.tires - self.tire_wear_rate * throttle)

                position = self.position + self.speed
                if position >= self.track_length:
                    lap = self.lap + 1

                    # Lap finish bonus
                    reward += self.best_time / self.lap_time

                    # If we have finished the race
                    if lap >= self.num_laps:
                        # Progress bonus
                        reward += (self.track_length - self.position) / self.track_length

                        self.position = 0
                        terminated = True
                        finish = True

                    else:
                        self.lap = lap
                        self.lap_time = 0

                        # If we have finished a lap and pitting
                        if pit_call:
                            # Progress bonus
                            reward += (self.track_length - self.position) / self.track_length

                            self.position = 0
                            self.speed = 0
                            self.in_pit = True
                            self.refuel_units = self.fuel_capacity * refuel_amount
                            self.change_tires = change_tires
                            self.pit_time = self.base_pit_stop_time + \
                                math.ceil(self.refuel_time_per_unit * self.refuel_units) + \
                                self.tire_change_time * int(self.change_tires)

                        # If we have finished a lap and continuing without pitting
                        else:
                            # Progress bonus
                            reward += (self.speed - self.min_speed) / self.track_length

                            self.position = position % self.track_length
                            self.lap_time += 1

                # If we are in the middle of a lap
                else:
                    # Progress bonus
                    reward += (self.speed - self.min_speed) / self.track_length

                    self.position = position

        self.render()

        return self._get_obs(), \
            reward, \
            terminated, \
            False, \
            self._get_info(finish)

    def render(self):
        if self.render_mode is None:
            return
        
        if self.render_mode == "human":
            self._render_human()
        
        if self.render_mode == "terminal":
            self._render_terminal()

    def _render_terminal(self):
        print("-" * 25)
        print(f"Lap       : {self.lap + 1}/{self.num_laps}")
        print(f"Completed : {(self.position / self.track_length * 100):.2f}%")
        print(f"Time      : {self.lap_time}")
        print(f"Speed     : {self.speed:.2f}")
        print(f"Fuel      : {(self.fuel / self.fuel_capacity * 100):.2f}%")
        print(f"Tires     : {(self.tires * 100):.2f}%")
        if self.in_pit:
            print(f"In Pit    : {self.pit_timer}/{self.pit_time}")
        print("-" * 25)

    def _render_human(self):
        # Init pygame once
        if self.screen is None:
            pygame.init()

            self.screen_size = (800, 600)
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Race Pit Strategy")

            self.clock = pygame.time.Clock()

            # Track ellipse parameters
            self.center = (self.screen_size[0] // 2, self.screen_size[1] // 2)
            self.track_radius = (350, 200) # Major/minor radii

        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Green background
        self.screen.fill((50, 150, 50))

        # Draw track
        pygame.draw.ellipse(
            self.screen,
            (90, 85, 80),
            (
                self.center[0] - self.track_radius[0],
                self.center[1] - self.track_radius[1],
                self.track_radius[0] * 2,
                self.track_radius[1] * 2
            ),
            width=0
        )

        # Draw inner grass
        pygame.draw.ellipse(
            self.screen,
            (50, 150, 50),
            (
                self.center[0] - (self.track_radius[0] - 40),
                self.center[1] - (self.track_radius[1] - 40),
                (self.track_radius[0] - 40) * 2,
                (self.track_radius[1] - 40) * 2
            ),
            width=0
        )

        # Compute car position on ellipse
        frac = (self.position % self.track_length) / self.track_length
        angle = -2 * math.pi * frac - math.pi / 2
        x = self.center[0] + math.cos(angle) * (self.track_radius[0] - 10)
        y = self.center[1] + math.sin(angle) * (self.track_radius[1] - 10)
        # Offset car when in pit
        if self.in_pit:
            y += 25

        # Draw car
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            (int(x), int(y)),
            10
        )

        font = pygame.font.SysFont(name="arial", size=24)
        font_bold = pygame.font.SysFont(name="arial", size=24, bold=True)

        # Blit current lap and lap time
        lap_text = font_bold.render(f"Lap: {self.lap + 1}/{self.num_laps}", True, (255, 255, 255))
        self.screen.blit(lap_text, (10, 10))
        lap_time_text = font_bold.render(f"Time: {self.lap_time}", True, (255, 255, 255))
        self.screen.blit(lap_time_text, (10, 40))

        # Blit pit info if pitting; speed, fuel and tire conditions otherwise
        if self.in_pit:
            speed_text = font_bold.render(f"In Pit: {self.pit_timer}/{self.pit_time}", True, (255, 255, 255))
            self.screen.blit(speed_text, (10, self.screen_size[1] - 50))
        else:
            speed_text = font.render(f"Speed: {self.speed:.2f}", True, (255, 255, 255))
            self.screen.blit(speed_text, (10, self.screen_size[1] - 90))
            fuel_text = font.render(f"Fuel: {(self.fuel / self.fuel_capacity * 100):.2f}%", True, (255, 255, 255))
            self.screen.blit(fuel_text, (10, self.screen_size[1] - 65))
            tire_text = font.render(f"Tires: {(self.tires * 100):.2f}%", True, (255, 255, 255))
            self.screen.blit(tire_text, (10, self.screen_size[1] - 40))

        pygame.display.flip()
        self.clock.tick(self.metadata.get("render_fps", 30))