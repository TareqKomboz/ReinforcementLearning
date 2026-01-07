import numpy as np
import matplotlib.pyplot as plt
import random


class GaussianBandit:
    def __init__(self):
        self._arm_means = np.random.uniform(0., 1., 10)  # Sample some means
        self.n_arms = len(self._arm_means)
        self.rewards = []
        self.total_played = 0

    def reset(self):
        self.rewards = []
        self.total_played = 0

    def play_arm(self, a):
        reward = np.random.normal(self._arm_means[a], 1.)  # Use sampled mean and covariance of 1.
        self.total_played += 1
        self.rewards.append(reward)
        return reward


def greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # Init variables by playing each arm once
    for current_arm in possible_arms:
        rewards[current_arm] += bandit.play_arm(current_arm)
        n_plays[current_arm] += 1
        Q[current_arm] = rewards[current_arm] / n_plays[current_arm]

    # Main loop
    while bandit.total_played < timesteps:
        # Do greedy action selection
        greedy_arm = 0
        for current_arm in possible_arms:
            if Q[current_arm] > Q[greedy_arm]:
                greedy_arm = current_arm

        # Update the variables (rewards, n_plays, Q) for the selected arm
        rewards[greedy_arm] += bandit.play_arm(greedy_arm)
        n_plays[greedy_arm] += 1
        Q[greedy_arm] = rewards[greedy_arm] / n_plays[greedy_arm]


def epsilon_greedy(bandit, timesteps):
    epsilon = 0.1

    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # Init variables by playing each arm once
    for current_arm in possible_arms:
        rewards[current_arm] += bandit.play_arm(current_arm)
        n_plays[current_arm] += 1
        Q[current_arm] = rewards[current_arm] / n_plays[current_arm]

    # Main loop
    while bandit.total_played < timesteps:
        if random.random() >= epsilon:
            # Do greedy action selection
            greedy_arm = 0
            for current_arm in possible_arms:
                if Q[current_arm] > Q[greedy_arm]:
                    greedy_arm = current_arm
            selected_arm = greedy_arm
        else:
            # Do random action selection
            random_arm = random.choice(possible_arms)
            selected_arm = random_arm

        # Update the variables (rewards, n_plays, Q) for the selected arm
        rewards[selected_arm] += bandit.play_arm(selected_arm)
        n_plays[selected_arm] += 1
        Q[selected_arm] = rewards[selected_arm] / n_plays[selected_arm]


def main():
    n_episodes = 10000  # to reduce noise
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        greedy(b, n_timesteps)
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        epsilon_greedy(b, n_timesteps)
        rewards_egreedy += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy)))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig('bandit_strategies.eps')
    plt.show()


if __name__ == "__main__":
    main()
