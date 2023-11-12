"""Test to make sure the DLP and RL are using the same model."""

from env.mrp_env_rccc import MachineReplacement
from experiments.configs.base import params
from utils.mrp import MRPData


def test_mdp_model():
    """Test whether RL and DLP model match under different parameter combinations."""

    for grp in [2, 3, 4, 5, 6]:
        for st in [2, 3, 4]:
            for prob in [0.5, 0.5, 0.7, 0.8]:
                params.num_groups = grp
                params.num_states = st
                params.prob_remain = prob
                # set up the environment
                env = MachineReplacement(
                    num_groups=params.num_groups,
                    num_states=params.num_states,
                    num_actions=params.num_actions,
                    prob_remain=params.prob_remain,
                    num_steps=params.len_episode,
                    encoding_int=True,
                )
                # solve with given policy
                mrp_data = MRPData(
                    num_groups=params.num_groups,
                    num_states=params.num_states,
                    num_actions=params.num_actions,
                    prob_remain=params.prob_remain,
                )

                # test: check rewards for all state and action pair
                for action in range(mrp_data.num_global_actions):
                    for state in range(mrp_data.num_global_states):
                        env.observations = state
                        next_state, reward_list, done, info = env.step(action)
                        if str(reward_list) != str(
                            mrp_data.global_costs[state, action, :]
                        ):
                            print(f"S{state}A{action}S'{next_state}")
                            print("reward list:", reward_list)
                            print(
                                "global costs:",
                                mrp_data.global_costs[next_state, action, :],
                            )


test_mdp_model()
