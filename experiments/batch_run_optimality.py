import matplotlib.pyplot as plt
import pandas as pd

from env.mrp_simulation import PropCountSimMDPEnv
from experiments.configs.base import params
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from utils.callbacks import SaveOnBestTrainingRewardCallback
from utils.common import update_params
from utils.plots import moving_average

params.update({"num_episodes": 300, "len_episode": 300})
algorithms = [PPO, SAC, TD3]
FILE_OUT = True
GROUPS = [10, 50, 100]
BUDGET_PROPS = [0.1, 0.2, 0.5]


def train_agent(algorithm, params):
    env = PropCountSimMDPEnv(
        machine_range=params.machine_range
        if params.machine_range
        else [params.num_groups, params.num_groups],
        resource_range=params.resource_range
        if params.resource_range
        else [params.budget, params.budget],
        num_states=params.num_states,
        len_episode=params.len_episode,
        cost_types_operation=params.cost_type_operation,
        cost_types_replace=params.cost_type_replace,
        force_to_use_all_resources=params.force_to_use_all_resources,
    )
    env = Monitor(env, params.log_dir)
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=params.len_episode * 20,
        log_dir=params.log_dir,
        model_name=f"{algorithm.__name__.lower()}_{params.identifier}",
    )
    total_timesteps = params.num_episodes * params.len_episode
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(
    #     mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    # )
    model = algorithm(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,
        gamma=params.gamma,
    )
    # action_noise=action_noise)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    training_rewards = env.env.training_rewards[: params.num_episodes]
    return model, training_rewards


for group in GROUPS:
    params.num_groups = group
    for budget in BUDGET_PROPS:
        params.budget = int(budget * group)
        params = update_params(params, group, params.budget)

        # train the agents
        for algorithm in algorithms:
            model, training_rewards = train_agent(algorithm, params)
            file_name = f"experiments/tmp/learning_reward_{algorithm.__name__.lower()}_{params.identifier}.csv"
            if FILE_OUT:
                pd.DataFrame(training_rewards).to_csv(file_name)
            plt.plot(
                moving_average(training_rewards, window=10), label=algorithm.__name__
            )

        # plot figures
        plt.legend()
        plt.xlabel("Episodes")
        plt.ylabel("GGF Expected Returns")
        plt.title("Learning Curves (Smoothed)")
        plt.savefig(f"experiments/tmp/learning_reward_{params.identifier}.png")
        plt.show()
