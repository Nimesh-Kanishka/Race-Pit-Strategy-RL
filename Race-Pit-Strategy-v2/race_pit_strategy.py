import functools
import math
import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
import pygame


def env(
    render_mode: str | None = None,
    max_episode_steps: int | None = 2500,
    **kwargs
):
    """
    Wrap the environment in wrappers by default.
    """

    env = raw_env(render_mode, max_episode_steps, **kwargs)

    # Provides a wide vareity of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)

    return env


def raw_env(
    render_mode: str | None = None,
    max_episode_steps: int | None = 2500,
    **kwargs
):
    """
    To support the AEC API, use the parallel_to_aec function
    to convert from a ParallelEnv to an AEC env.
    """

    env = parallel_env(render_mode, max_episode_steps, **kwargs)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "render_fps": 30, "name": "race_pit_strategy_v2"}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int | None = 2500,
        num_cars: int = 10,
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
        self.max_episode_steps = max_episode_steps
        self.possible_agents = ["car_" + str(id) for id in range(num_cars)]

        # Environment parameters
        self.num_cars = num_cars
        self.num_laps = num_laps
        self.track_length = track_length
        self.race_distance = num_laps * track_length
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

        # Pygame variables
        self.screen_size = None
        self.screen = None
        self.clock = None
        self.center = None
        self.track_radius = None

    # Define observation space.
    # lru_cache allows observation space to be memoized, reducing clock cycles required to get each agent's space.
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Observation space consists of 10 observations: Current lap (Normalized), Position on track
        # (Normalized), Position in race (Normalized), Distance to the nearest car in front (Normalized),
        # Distance to the nearest car behind (Normalized), Speed (Normalized), Fuel left (Normalized),
        # Tire left(Normalized), Whether in pitlane or not, Pit timer (Normalized).
        # Each observation can take any value in the range [0, 1].
        return Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32
        )

    # Define action space.
    # lru_cache allows action space to be memoized, reducing clock cycles required to get each agent's space.
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Action space consists of 4 actions: Throttle, Pit call, Refuel amount, Change tires.
        # Each action can take any value in the range [0, 1].
        return Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )        

    def reset(self, seed=None, options=None):
        self.episode_steps = 0
        self.agents = self.possible_agents[:]

        # State variables
        self.lap = {agent: 0 for agent in self.agents}
        self.position = {agent: 0.0 for agent in self.agents}
        self.distance = {agent: 0.0 for agent in self.agents}
        self.place = {agent: 1 for agent in self.agents}
        self.lap_time = {agent: 0 for agent in self.agents}
        self.speed = {agent: 0.0 for agent in self.agents}
        self.fuel = {agent: self.fuel_capacity for agent in self.agents}
        self.tires = {agent: 1.0 for agent in self.agents}
        self.in_pit = {agent: False for agent in self.agents}
        self.refuel_units = {agent: 0 for agent in self.agents}
        self.change_tires = {agent: False for agent in self.agents}
        self.pit_time = {agent: 0 for agent in self.agents}
        self.pit_timer = {agent: 0 for agent in self.agents}

        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}

        self.render()

        return observations, infos

    def _get_obs(self):
        obs = {}

        for agent in self.agents:
            # Current lap (Normalized)
            lap = self.lap[agent] / self.num_laps
            # Position on track (Normalized)
            position = self.position[agent] / self.track_length
            # Position in race (Normalized)
            place = 1 - (self.place[agent] - 1) / (self.num_cars - 1)
            # Distances to the nearest cars in front and behind (Normalized)
            distance_in_front = distance_behind = 1.0
            for a, p in self.place.items():
                if p == self.place[agent] - 1:
                    distance_in_front = (self.distance[a] - self.distance[agent]) / self.race_distance
                elif p == self.place[agent] + 1:
                    distance_behind = (self.distance[agent] - self.distance[a]) / self.race_distance
            # Speed (Normalized)
            speed = self.speed[agent] / self.max_speed
            # Fuel left (Normalized)
            fuel = self.fuel[agent] / self.fuel_capacity
            # Tire left (Normalized)
            tires = self.tires[agent]
            # Whether in pitlane (1) or not (0)
            in_pit = float(self.in_pit[agent])
            # Pit timer (Normalized)
            pit_timer = self.pit_timer[agent] / self.pit_time[agent] if self.in_pit[agent] else 0.0

            obs[agent] = np.array(
                [
                    lap,
                    position,
                    place,
                    distance_in_front,
                    distance_behind,
                    speed,
                    fuel,
                    tires,
                    in_pit,
                    pit_timer
                ],
                dtype=np.float32
            )

        return obs

    def step(self, actions):
        self.episode_steps += 1
        env_truncation = self.episode_steps >= self.max_episode_steps if self.max_episode_steps is not None else False
        truncations = {agent: env_truncation for agent in self.agents}

        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        
        # Update the position of each car
        for agent in self.agents:
            self.lap_time[agent] += 1

            # If we are in pitlane
            if self.in_pit[agent]:
                self.pit_timer[agent] += 1

                if self.pit_timer[agent] == self.pit_time[agent]:
                    self.fuel[agent] = min(self.fuel[agent] + self.refuel_units[agent], self.fuel_capacity)
                    self.tires[agent] = 1.0 if self.change_tires[agent] else self.tires[agent]
                    self.in_pit[agent] = False
                    self.refuel_units[agent] = 0
                    self.change_tires[agent] = False
                    self.pit_time[agent] = 0
                    self.pit_timer[agent] = 0

            else:
                # If we have run out of fuel or worn out tires
                if self.fuel[agent] == 0 or self.tires[agent] == 0:
                    self.speed[agent] = 0.0

                    # Penalty for running out of fuel or worn out tires
                    rewards[agent] -= 25.0

                    terminations[agent] = True

                else:
                    max_speed = self.max_speed * \
                        (((0.55 * self.tires[agent] - 1.75) * self.tires[agent] + 1.9) * self.tires[agent] + 0.3)
                    throttle = actions[agent][0]

                    self.speed[agent] = min(max_speed * throttle,
                                            self.fuel[agent] * max_speed / self.fuel_consumption_rate,
                                            self.tires[agent] * max_speed / self.tire_wear_rate)
                    
                    self.fuel[agent] = max(0.0, self.fuel[agent] - self.fuel_consumption_rate * throttle)
                    self.tires[agent] = max(0.0, self.tires[agent] - self.tire_wear_rate * throttle)

                    position = self.position[agent] + self.speed[agent]
                    if position >= self.track_length:
                        # Lap finish bonus
                        rewards[agent] += self.best_time / self.lap_time[agent]

                        self.lap[agent] += 1
                        self.lap_time[agent] = 0

                        # If we have finished the race
                        if self.lap[agent] == self.num_laps:
                            distance = self.track_length - self.position[agent]
                            self.position[agent] += distance
                            self.distance[agent] += distance

                            # Progress bonus
                            rewards[agent] += distance / self.track_length

                            terminations[agent] = True

                        # If we have finished a lap and pitting
                        elif bool(np.round(actions[agent][1])):
                            distance = self.track_length - self.position[agent]
                            self.position[agent] = 0
                            self.distance[agent] += distance

                            # Progress bonus
                            rewards[agent] += distance / self.track_length

                            self.speed[agent] = 0.0
                            self.in_pit[agent] = True
                            self.refuel_units[agent] = self.fuel_capacity * actions[agent][2]
                            self.change_tires[agent] = bool(np.round(actions[agent][3]))
                            self.pit_time[agent] = self.base_pit_stop_time + \
                                math.ceil(self.refuel_time_per_unit * self.refuel_units[agent]) + \
                                self.tire_change_time * int(self.change_tires[agent])

                        # If we have finished a lap and continuing without pitting
                        else:
                            self.position[agent] = position % self.track_length
                            self.distance[agent] += self.speed[agent]
                            self.lap_time[agent] += 1

                            # Progress bonus
                            rewards[agent] += (self.speed[agent] - self.min_speed) / self.track_length

                    # If we are in the middle of a lap
                    else:
                        self.position[agent] = position
                        self.distance[agent] += self.speed[agent]

                        # Progress bonus
                        rewards[agent] += (self.speed[agent] - self.min_speed) / self.track_length

        # Calculate the place of each car
        sorted_agents = sorted(self.agents, key=lambda agent: self.distance[agent], reverse=True)

        prev_dist = None
        current_rank = 0

        for i, agent in enumerate(sorted_agents):
            dist = self.distance[agent]
            # If the distance is less than the previous distance, we have a new rank,
            # so we increase current rank to i + 1
            if dist != prev_dist:
                current_rank = i + 1
                prev_dist = dist
            # Otherwise, there is a tie, so current_rank stays the same

            self.place[agent] = current_rank

            # Penalty for being behind the lead car
            rewards[agent] -= (current_rank - 1) / (self.num_cars - 1) / self.best_time

        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}

        self.render()

        # Remove any agents that have terminated or been truncated
        self.agents = [
            agent for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]

        return observations, rewards, terminations, truncations, infos
    
    def close(self):
        if self.screen is not None:
            pygame.quit()

            self.screen_size = None
            self.screen = None
            self.clock = None
            self.center = None
            self.track_radius = None

    def render(self):
        if self.render_mode is None:
            return
        
        self._render_human()

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
                self.close()
                return

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

        # Define font for displaying agent information
        font = pygame.font.SysFont(name="arial", size=18)

        # Draw headings
        headings_text = font.render("Rank    Lap    Time    Speed    Fuel        Tires", True, (255, 255, 255))
        self.screen.blit(headings_text, (10, 2))

        for agent in self.agents:
            # Compute car position on ellipse
            frac = (self.position[agent] % self.track_length) / self.track_length
            angle = -2 * math.pi * frac - math.pi / 2
            x = self.center[0] + math.cos(angle) * (self.track_radius[0] - 10)
            y = self.center[1] + math.sin(angle) * (self.track_radius[1] - 10)
            # Offset car when in pit
            if self.in_pit[agent]:
                y += 25

            # Generate a unique color for each car
            color = (
                (hash(agent + "red") % 256),
                (hash(agent + "green") % 256),
                (hash(agent + "blue") % 256)
            )

            # Draw car as a circle
            pygame.draw.circle(
                self.screen,
                color,
                (int(x), int(y)),
                10
            )

            # Display agent information
            agent_info = f"  {self.place[agent]:02d}       {(self.lap[agent] + 1):02d}       {self.lap_time[agent]:02d}       "
            if self.in_pit[agent]:
                agent_info += f"         In Pit: {self.pit_timer[agent]}/{self.pit_time[agent]}"
            else:
                agent_info += f"{self.speed[agent]:05.2f}   {(self.fuel[agent] / self.fuel_capacity * 100):05.2f}%   {(self.tires[agent] * 100):05.2f}%"                
            agent_info_text = font.render(agent_info, True, color)
            self.screen.blit(agent_info_text, (10, 20 * self.place[agent]))

        pygame.display.flip()
        self.clock.tick(self.metadata.get("render_fps", 30))