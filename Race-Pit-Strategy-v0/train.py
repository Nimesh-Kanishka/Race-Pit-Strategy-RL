import gymnasium as gym
from race_pit_strategy import RacePitStrategyEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement


def make_env(render: bool = False):
    return gym.make("RacePitStrategy-v0",
                    render_mode="human" if render else None)


def train(
    num_envs: int = 8,
    device: str = "cpu"
):
    env = make_vec_env(make_env,
                       n_envs=num_envs,
                       vec_env_cls=SubprocVecEnv)

    model = PPO(policy=MultiInputActorCriticPolicy,
                env=env,
                tensorboard_log="Race_Pit_Strategy/Race_Pit_Strategy_v0/Logs",
                verbose=0,
                device=device)
    
    eval_callback = EvalCallback(env,
                                 callback_after_eval=StopTrainingOnNoModelImprovement(max_no_improvement_evals=50,
                                                                                      min_evals=200,
                                                                                      verbose=1),
                                 eval_freq=10_000 // num_envs,
                                 best_model_save_path="Race_Pit_Strategy/Race_Pit_Strategy_v0/Models/PPO",
                                 verbose=1)

    model.learn(total_timesteps=1e9,
                callback=eval_callback,
                reset_num_timesteps=False)
    

def test(
    total_episodes: int,
    render: bool = True,
    device: str = "cpu"
):
    env = make_env(render=render)

    model = PPO.load(path=f"Race_Pit_Strategy/Race_Pit_Strategy_v0/Models/PPO/best_model",
                     env=env,
                     device=device)

    for episode in range(total_episodes):
        observation, _ = env.reset()
        episode_steps = 0
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(observation=observation,
                                      deterministic=True)

            observation, reward, terminated, truncated, _ = env.step(action)

            episode_steps += 1
            episode_reward += reward
            done = terminated or truncated

        print(f"Episode: {episode + 1}/{total_episodes} - Steps: {episode_steps} - Reward: {episode_reward:.2f}")


if __name__ == "__main__":
    train()

    test(total_episodes=1)
