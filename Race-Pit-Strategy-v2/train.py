import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import race_pit_strategy
from custom_agents.pit_strategy_agent import PitStrategyAgent
from custom_agents.random_agent import RandomAgent
from custom_agents.custom_pit_agent import CustomPitAgent


def train(
    env_fn,
    total_timesteps: int,
    num_vec_envs: int = 8,
    verbose: bool = False,
    **env_kwargs
):
    # Train a single model to play as each agent in a Parallel environment
    env = env_fn.parallel_env(**env_kwargs)

    # Add black death wrapper so the number of agents stays constant
    # MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True
    env = ss.black_death_v3(env)

    env.reset()

    print(f"--- Starting Training (env={str(env.metadata['name'])}, total_timesteps={total_timesteps}) ---")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs, num_cpus=1, base_class="stable_baselines3")

    # Create folders to save logs and models if they do not exist
    os.makedirs("Race_Pit_Strategy/Race_Pit_Strategy_v2/Logs", exist_ok=True)
    os.makedirs("Race_Pit_Strategy/Race_Pit_Strategy_v2/Models", exist_ok=True)

    model = PPO(
        policy=MlpPolicy,
        env=env,
        tensorboard_log="Race_Pit_Strategy/Race_Pit_Strategy_v2/Logs",
        verbose=int(verbose),
        device="cpu"
    )

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False,
        progress_bar=True
    )

    model.save("Race_Pit_Strategy/Race_Pit_Strategy_v2/Models/model_final_ppo")

    print(f"--- Finished Training ---")

    env.close()


def eval(
    env_fn,
    num_episodes: int,
    render: bool = False,
    other_agents: list[PitStrategyAgent] = [],
    **env_kwargs
):
    # num_cars must at least be the number of other agents + 1 (the PPO agent)
    if "num_cars" in env_kwargs:
        num_cars = env_kwargs.pop("num_cars")
        num_cars = len(other_agents) + 1 if num_cars <= len(other_agents) else num_cars
    else:
        num_cars = len(other_agents) + 1

    env = env_fn.env(render_mode="human" if render else None, num_cars=num_cars, **env_kwargs)

    print(f"--- Starting Evaluation (env={str(env.metadata['name'])}, num_games={num_episodes}, render={render}) ---")

    model = PPO.load(
        path="Race_Pit_Strategy/Race_Pit_Strategy_v2/Models/model_final_ppo",
        device="cpu"
    )

    total_length_per_agent = {agent: 0 for agent in env.possible_agents}
    total_reward_per_agent = {agent: 0.0 for agent in env.possible_agents}

    # We train using the Parallel API but evaluate using the AEC API.
    # SB3 models are designed for single-agent settings, we get around
    # this by using the same model for every agent.
    for _ in range(num_episodes):
        env.reset()

        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()

            total_reward_per_agent[agent] += reward

            if termination or truncation:
                action = None
            else:
                total_length_per_agent[agent] += 1
                # Find the index of the agent in the possible_agents list
                agent_id = env.possible_agents.index(agent)
                # For the first len(other_agents) agents, we get actions by calling
                # the get_action method of the respective agent in other_agents list
                if agent_id < len(other_agents):
                    action = other_agents[agent_id].get_action(observation)
                # For the last agent/s we use the trained model (PPO)
                else:
                    action = model.predict(observation, deterministic=True)[0]
                    
            env.step(action)

    env.close()

    # Calculate average episode length and reward for each agent
    avg_length_per_agent = {
        agent: total_length_per_agent[agent] / num_episodes
        for agent in total_length_per_agent
    }
    avg_reward_per_agent = {
        agent: total_reward_per_agent[agent] / num_episodes
        for agent in total_reward_per_agent
    }

    print("-" * 35)
    print("Agent  |  Avg Length  |  Avg Reward")
    print("-" * 35)
    for agent in env.possible_agents:
        print(f"{agent}  |     {avg_length_per_agent[agent]:04.0f}     |    {avg_reward_per_agent[agent]:.2f}")
    print("-" * 35)

    print(f"--- Finished Evaluation ---")


if __name__ == "__main__":
    env_fn = race_pit_strategy
    
    # Train a model
    # Training for 20M timesteps takes ~90 minutes on my laptop (AMD Ryzen 7 7745HX CPU)
    train(env_fn, total_timesteps=20_480_000)

    # We will evaluate the trained agent against a random agent and a custom agent acting on pre-defined logic
    other_agents = [
        RandomAgent(),
        CustomPitAgent(),
    ]

    # Evaluate 10 episodes
    eval(env_fn, num_episodes=10, render=False, other_agents=other_agents, max_episode_steps=10_000)

    # Watch an episode
    eval(env_fn, num_episodes=1, render=True, other_agents=other_agents, max_episode_steps=10_000)
