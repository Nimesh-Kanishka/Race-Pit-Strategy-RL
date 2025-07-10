import functools
import math
import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
import pygame


def env(
    render_mode: str | None = None,
    max_episode_steps: int | None = 5000,
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
    max_episode_steps: int | None = 5000,
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
    metadata = {"render_modes": ["human"], "render_fps": 30, "name": "race_pit_strategy_v3"}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int | None = 5000,
        num_cars: int = 10,
        num_laps: int = 50,
        track_length: float = 2500.0,
        max_speed: float = 75.0,
        max_speed_under_caution: float = 20.0,
        fuel_capacity: float = 100.0,
        fuel_consumption_rate: float = 0.5,
        tire_wear_rate: float = 0.0015,
        base_pit_stop_time: int = 15,
        refuel_time_per_unit: float = 0.06,
        tire_change_time: int = 10,
        caution_probability: float = 0.001
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
        self.max_speed_under_caution = max_speed_under_caution
        self.fuel_capacity = fuel_capacity
        self.fuel_consumption_rate = fuel_consumption_rate
        self.tire_wear_rate = tire_wear_rate
        self.base_pit_stop_time = base_pit_stop_time
        self.refuel_time_per_unit = refuel_time_per_unit
        self.tire_change_time = tire_change_time
        self.caution_probability = caution_probability
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
        # Observation space consists of 11 observations: Current lap (Normalized), Position on track
        # (Normalized), Position in race (Normalized), Distance to the nearest car in front (Normalized),
        # Distance to the nearest car behind (Normalized), Speed (Normalized), Fuel left (Normalized),
        # Tire left(Normalized), Whether or not under caution, Whether or not in pitlane, Pit timer (Normalized).
        # Each observation can take any value in the range [0, 1].
        return Box(
            low=0.0,
            high=1.0,
            shape=(11,),
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
        self.distance_in_front = {agent: None for agent in self.agents}
        self.distance_behind = {agent: None for agent in self.agents}
        self.lap_time = {agent: 0 for agent in self.agents}
        self.speed = {agent: 0.0 for agent in self.agents}
        self.fuel = {agent: self.fuel_capacity for agent in self.agents}
        self.tires = {agent: 1.0 for agent in self.agents}
        self.in_pit = {agent: False for agent in self.agents}
        self.refuel_units = {agent: 0 for agent in self.agents}
        self.change_tires = {agent: False for agent in self.agents}
        self.pit_time = {agent: 0 for agent in self.agents}
        self.pit_timer = {agent: 0 for agent in self.agents}
        self.under_caution = False
        self.laps_under_caution = 0
        # We will be penalizing the agent if the speed is below a certain threshold. Otherwise the agent
        # may stop on the track and wait for the simulation to end to minimize other penalties.
        self.min_speed = self.max_speed / 1.15
        self.leader = self.agents[0]

        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}

        self.render()

        return observations, infos

    def _get_obs(self):
        return {
            agent: np.array(
                [
                    self.lap[agent] / self.num_laps,
                    self.position[agent] / self.track_length,
                    1 - (self.place[agent] - 1) / (self.num_cars - 1),
                    self.distance_in_front[agent] / self.race_distance if self.distance_in_front[agent] is not None else 1.0,
                    self.distance_behind[agent] / self.race_distance if self.distance_behind[agent] is not None else 1.0,
                    self.speed[agent] / self.max_speed,
                    self.fuel[agent] / self.fuel_capacity,
                    self.tires[agent],
                    float(self.under_caution),
                    float(self.in_pit[agent]),
                    self.pit_timer[agent] / self.pit_time[agent] if self.in_pit[agent] else 0.0
                ],
                dtype=np.float32
            ) for agent in self.agents
        }

    def step(self, actions):
        self.episode_steps += 1
        env_truncation = self.episode_steps >= self.max_episode_steps if self.max_episode_steps is not None else False
        truncations = {agent: env_truncation for agent in self.agents}

        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}

        # Check if any car not in pits has run out of fuel or worn out tires
        # If so, that agent will be penalized and terminated and the race will go under caution
        for agent in self.agents:
            if (self.fuel[agent] == 0 or self.tires[agent] == 0) and not self.in_pit[agent]:
                self.speed[agent] = 0.0

                # Penalty for running out of fuel or worn out tires
                rewards[agent] -= 50.0

                terminations[agent] = True

                # As a car has stopped on track, the race will go under caution
                if not self.under_caution:
                    self.under_caution = True
                    self.laps_under_caution = np.random.randint(low=3, high=5)
                    self.min_speed = self.max_speed_under_caution

        # Even if there are no cars stopped on track, the race will occasionally go under caution for randomness
        if not self.under_caution and np.random.random() < self.caution_probability:
            self.under_caution = True
            self.laps_under_caution = np.random.randint(low=3, high=5)
            self.min_speed = self.max_speed_under_caution
        
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
                throttle = actions[agent][0]

                tires = self.tires[agent]
                max_speed = self.max_speed * ((0.55 * tires - 1.75) * tires + 1.9) * tires + 0.3

                if self.under_caution and max_speed > self.max_speed_under_caution and (
                    self.laps_under_caution > 2 or
                    (self.position[self.leader] - self.position[agent]) % self.track_length < max_speed
                ):
                    max_speed_possible = self.max_speed_under_caution
                else:
                    max_speed_possible = max_speed

                speed = min(max_speed_possible * throttle,
                            self.fuel[agent] * max_speed / self.fuel_consumption_rate,
                            self.tires[agent] * max_speed / self.tire_wear_rate)
                
                self.speed[agent] = speed
                self.fuel[agent] = max(0.0, self.fuel[agent] - self.fuel_consumption_rate * speed / max_speed)
                self.tires[agent] = max(0.0, self.tires[agent] - self.tire_wear_rate * speed / max_speed)

                position = self.position[agent] + self.speed[agent]
                if position >= self.track_length:
                    # If the race leader has crossed the finish line, we will have finished a lap under caution
                    if self.under_caution and agent == self.leader:
                            self.laps_under_caution -= 1
                            if self.laps_under_caution <= 0:
                                self.under_caution = False
                                self.min_speed = self.max_speed / 1.15

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
                        rewards[agent] += (distance - self.min_speed) / self.track_length

                        terminations[agent] = True

                    # If we have finished a lap and pitting
                    elif bool(np.round(actions[agent][1])):
                        distance = self.track_length - self.position[agent]
                        self.position[agent] = 0
                        self.distance[agent] += distance

                        # Progress bonus
                        rewards[agent] += (distance - self.min_speed) / self.track_length

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

                        # Penalty for stopping on track
                        if self.speed[agent] < 10.0:
                            rewards[agent] -= 0.05

                # If we are in the middle of a lap
                else:
                    self.position[agent] = position
                    self.distance[agent] += self.speed[agent]

                    # Progress bonus
                    rewards[agent] += (self.speed[agent] - self.min_speed) / self.track_length

                    # Penalty for stopping on track
                    if self.speed[agent] < 10.0:
                        rewards[agent] -= 0.05

        # Calculate the place of each car
        sorted_agents = sorted(self.agents, key=lambda agent: self.distance[agent], reverse=True)

        self.leader = sorted_agents[0] if sorted_agents else None

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

            # The distances to the nearest cars in front and behind will be set to None by default
            # (if no car is in front or behind)
            self.distance_in_front[agent] = self.distance_behind[agent] = None

            # If there is a car in front, calculate the distance to it
            for j in range(i - 1, -1, -1):
                a = sorted_agents[j]
                if self.distance[a] > self.distance[agent]:
                    self.distance_in_front[agent] = self.distance[a] - self.distance[agent]
                    break

            # If there is a car behind, calculate the distance to it
            for j in range(i + 1, len(sorted_agents)):
                a = sorted_agents[j]
                if self.distance[a] < self.distance[agent]:
                    self.distance_behind[agent] = self.distance[agent] - self.distance[a]
                    break

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

        # Show whether the race is under green flag or caution (yellow)
        pygame.draw.rect(
            self.screen,
            (255, 230, 0) if self.under_caution else (0, 185, 0),
            pygame.Rect(self.screen_size[0] - 60, 10, 50, 30)
        )

        # Define font for displaying agent information
        font = pygame.font.SysFont(name="arial", size=18)

        # Draw headings
        headings_text = font.render("Rank    Car    Lap    Time    Speed    Fuel        Tires", True, (255, 255, 255))
        self.screen.blit(headings_text, (10, 2))

        for i, agent in enumerate(sorted(self.agents, key=lambda agent: self.place[agent])):
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
            agent_info = f"  {self.place[agent]:02d}        {self.possible_agents.index(agent):02d}     {(self.lap[agent] + 1):02d}       {self.lap_time[agent]:02d}       "
            if self.in_pit[agent]:
                agent_info += f"         In Pit: {self.pit_timer[agent]}/{self.pit_time[agent]}"
            else:
                agent_info += f"{self.speed[agent]:05.2f}   {(self.fuel[agent] / self.fuel_capacity * 100):05.2f}%   {(self.tires[agent] * 100):05.2f}%"                
            agent_info_text = font.render(agent_info, True, color)
            self.screen.blit(agent_info_text, (10, 20 * (i + 1)))

        pygame.display.flip()
        self.clock.tick(self.metadata.get("render_fps", 30))