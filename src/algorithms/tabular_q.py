import numpy as np
import pandas as pd


class QAgent:
    def __init__(
        self,
        env,
        num_states,
        num_actions,
        num_groups,
        alpha,
        decaying_factor,
        epsilon,
        gamma,
    ):
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_groups = num_groups
        self.ggi = env.ggi
        self.q_table = np.random.uniform(
            low=3000, high=3100, size=(num_states, num_actions)
        )
        self.q_table_visit_freq = np.zeros(shape=(num_states, num_actions))
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.decaying_factor = decaying_factor
        self.alpha = alpha
        self.gamma = gamma
        self.lr_decay_schedule = []

    def get_action(self, state):
        # We use epsilon-greedy to get an action
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(self.q_table[state][:])
        return action

    def update_q_function(self, state, action, reward, state_next):
        # Update the Q function with linear function approximation
        # In this case, equivalent to the tabular Q-Learning
        q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + self.alpha * (
            reward + self.gamma * q_next - self.q_table[state, action]
        )
        self.q_table_visit_freq[state, action] += 1

    def get_policy(self):
        # We use greedy to decide an optimal policy
        state_action_pair = {}
        for state in range(self.num_states):
            state_action_pair[state] = np.argmax(self.q_table[state][:])
        return state_action_pair


def run_tabular_q(
    env,
    num_episodes=200,
    len_episode=1000,
    alpha=0.1,
    epsilon=0.2,
    decaying_factor=0.95,
    gamma=0.99,
):
    episode_rewards = []
    agent = QAgent(
        env=env,
        num_states=env.observation_space.n,
        num_actions=env.action_space.n,
        num_groups=env.mrp_data.n_group,
        alpha=alpha,
        epsilon=epsilon,
        decaying_factor=decaying_factor,
        gamma=gamma,
    )
    agent.lr_decay_schedule = np.linspace(alpha, 0, num_episodes)
    # Run 200 episodes
    for episode in range(1, num_episodes + 1):
        # Initialize environment
        state = env.reset()
        total_reward = 0
        for t in range(len_episode):
            # Get an action
            action = agent.get_action(state)
            # Move
            state_next, reward, done, _ = env.step(action)
            # Update Q table
            agent.update_q_function(state, action, reward, state_next)
            # Update observation
            state = state_next
            total_reward += (gamma ** t) * reward
        # We use a decaying epsilon-greedy method
        if agent.epsilon > 0.001:
            agent.epsilon = agent.epsilon * agent.decaying_factor
        agent.alpha = agent.lr_decay_schedule[episode - 1]
        # Display running rewards
        if episode % 20 == 0:
            print(
                f"Episode: {episode}; "
                f"Running Reward: {total_reward:.1f}; "
                f"Epsilon: {agent.epsilon:.3f}; "
                f"Learning Rate: {agent.alpha:.3f}"
            )
        episode_rewards.append(total_reward)
    state_action_pair = agent.get_policy()
    print("===== Q table =====")
    print(pd.DataFrame(agent.q_table))
    print("===== Q table visitation count =====")
    print(pd.DataFrame(agent.q_table_visit_freq))
    return state_action_pair, episode_rewards


# run the script as main
if __name__ == "__main__":
    import argparse

    import numpy as np
    import seaborn as sns

    from env.mrp_env import MachineReplace
    from experiments.configs.config_shared import prs_parser_setup

    # configuration parameters
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Tabular Q Learning.""",
    )
    prs = prs_parser_setup(prs)
    args, _ = prs.parse_known_args()

    # initialize the MRP environment
    env = MachineReplace(
        n_group=args.n_group,
        n_state=args.n_state,
        n_action=args.n_action,
        init_state=0,
        ggi=args.ggi,
    )

    tabular_q_policy, episode_rewards = run_tabular_q(
        env=env,
        num_episodes=500,
        len_episode=1500,
        alpha=0.1,
        epsilon=0.2,
        decaying_factor=0.99,
        gamma=0.90,
    )
