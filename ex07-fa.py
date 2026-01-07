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

def float_to_interval_index(observation, x_intervals, vel_intervals):
    for i, x in enumerate(x_intervals):
        if i == 0:
            continue
        if observation[0] < x:
            x_index = i
            break

    for i, vel in enumerate(vel_intervals):
        if i == 0:
            continue
        if observation[1] < vel:
            vel_index = i
            break

    return (x_index, vel_index)

def qlearning(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e3)):
    """ This is Q-Learning """

    interval_size = 20
    x_intervals = np.linspace(-1.2, 0.6, interval_size)
    vel_intervals = np.linspace(-0.07, 0.07, interval_size)
    Q = np.zeros((interval_size, interval_size, env.action_space.n))

    for _ in range(num_ep):
        observation = env.reset()
        (obs_x, obs_vel) = float_to_interval_index(observation, x_intervals, vel_intervals)
        done = False
        while not done:
            env.render()
            if np.random.uniform(low=0.0, high=1.0) < epsilon:
                a = np.random.randint(env.action_space.n)
            else:
                a = np.argmax(Q[obs_x, obs_vel])

            observation_, reward, done, _ = env.step(a)
            (obs_x_, obs_vel_) = float_to_interval_index(observation_, x_intervals, vel_intervals)
            Q[obs_x, obs_vel, a] += alpha * (reward + gamma * np.max(Q[obs_x_, obs_vel_]) - Q[obs_x, obs_vel, a])

            (obs_x, obs_vel) = (obs_x_, obs_vel_)

def main():
    env = gym.make('MountainCar-v0')
    env.reset()
    # random_episode(env)
    qlearning(env)
    env.close()


if __name__ == "__main__":
    main()
