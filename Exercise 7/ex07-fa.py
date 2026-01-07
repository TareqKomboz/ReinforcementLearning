import gym
import numpy as np
import matplotlib.pyplot as plt


def random_episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        env.render()
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("reward: ", reward)
        print("")
        if done:
            break


def q_learning_with_state_aggregation(env, alpha=0.8, eps=0.1, num_eps=10000):
    # Todo: Implement Q-learning by using state-aggregation (e.g., 20 intervals for x and 20 for ˙x). Plot the value function at regular intervals (e.g., every 20 episodes)
    Q = np.zeros((20, 20, 3))
    gamma = 0.99

    for i in range(num_eps):
        state = env.reset()
        while not done:
            if np.random.uniform(0, 1) < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
            observation, reward, done, info = env.step(action)
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[observation, :]) - Q[state, action])
            state = observation
        if i % 20 == 0:
            plot_value_function(Q, i)

    return Q


def plot_value_function(Q, i):
    x = np.arange(-1.2, 0.6, 0.1)
    y = np.arange(-0.07, 0.07, 0.01)

    plt.plot(x, y)


def plot(num_eps):



def main():
    env = gym.make('MountainCar-v0')
    # q_learning_with_state_aggregation(env=env, alpha=)
    random_episode(env)
    env.close()


if __name__ == "__main__":
    main()
