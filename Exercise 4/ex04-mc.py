import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v0')


def single_run_20():
    """ run the policy that sticks for >= 20 """
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    # It can be used for the subtasks
    # Use a comment for the print outputs to increase performance (only there as example)
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    states = []
    actions = []
    ret = 0.
    while not done:
        # print("observation:", obs)
        states.append(obs)
        if obs[0] >= 20:
            # print("stick")
            obs, reward, done, _ = env.step(0)  # step=0 for stick
            actions.append(0)
        else:
            # print("hit")
            obs, reward, done, _ = env.step(1)  # step=1 for hit
            actions.append(1)
        # print("reward:", reward, "\n")
        ret += reward  # Note that gamma = 1. in this exercise
    # print("final observation:", obs)
    return states, ret, actions


def policy_evaluation():
    """ Implementation of first-visit Monte Carlo prediction """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    V = np.zeros((10, 10, 2))
    returns = np.zeros((10, 10, 2))  # create a 3D array of lists
    maxiter = 10000  # use whatever number of iterations you want
    for i in range(maxiter):
        visits = np.zeros((10, 10, 2))
        states, ret, actions = single_run_20()  # generate an episode following the policy that sticks for >= 20
        g = 0
        for t in range(len(states) - 1, -1, -1):
            g += ret
            current_state = states[t]
            current_player_sum = current_state[0]
            current_dealer_card = current_state[1]
            current_usable_ace = current_state[2]
            current_usable_ace_index = int(current_usable_ace)

            if 12 <= current_player_sum <= 21:
                if visits[current_player_sum - 12, current_dealer_card - 1, current_usable_ace_index] == 0:
                    if t == 0 or not (current_state in states[0:t-1]):
                        returns[current_player_sum - 12, current_dealer_card - 1, current_usable_ace_index] += g
                    visits[current_player_sum - 12, current_dealer_card - 1, current_usable_ace_index] += 1
                    V[current_player_sum - 12, current_dealer_card - 1, current_usable_ace_index] = \
                        returns[current_player_sum - 12, current_dealer_card - 1, current_usable_ace_index] / \
                        visits[current_player_sum - 12, current_dealer_card - 1, current_usable_ace_index]
    return V


def monte_carlo_es():
    """ Implementation of Monte Carlo ES """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    pi = np.zeros((10, 10, 2))
    Q = np.ones((10, 10, 2, 2)) * 100  # recommended: optimistic initialization of Q
    returns = np.zeros((10, 10, 2, 2))
    visits = np.zeros((10, 10, 2, 2))
    maxiter = 1000000  # 100000000  # use whatever number of iterations you want
    for i in range(maxiter):
        states, ret, actions = single_run_exploring_starts(pi)  # generate an episode following the policy that sticks for >= 20 while exploring starts
        g = 0
        for t in range(len(states) - 1, -1, -1):
            g += ret
            current_state = states[t]
            current_player_sum = current_state[0]
            current_dealer_card = current_state[1]
            current_usable_ace = current_state[2]
            current_usable_ace_index = int(current_usable_ace)
            current_action = actions[t]

            if t == 0 or (not (current_state in states[0:t-1]) and not ((current_action in actions[0:t-1]))):
                returns[current_player_sum - 12, current_dealer_card - 1, current_usable_ace_index, current_action] += g
                visits[current_player_sum - 12, current_dealer_card - 1, current_usable_ace_index, current_action] += 1
                Q[current_player_sum - 12, current_dealer_card - 1, current_usable_ace_index, current_action] \
                    = returns[current_player_sum - 12, current_dealer_card - 1, current_usable_ace_index, current_action] / visits[current_player_sum - 12, current_dealer_card - 1, current_usable_ace_index, current_action]
                pi[current_player_sum - 12, current_dealer_card - 1, current_usable_ace_index] \
                    = np.argmax(Q[current_player_sum - 12, current_dealer_card - 1, current_usable_ace_index])
        if i % 100000 == 0:
            print("Iteration: " + str(i))
            print(pi[:, :, 0])
            print(pi[:, :, 1])


def single_run_exploring_starts(pi):
    """ run the policy pi for a single episode and
    return the states, rewards and actions """
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    states = []
    actions = []
    ret = 0.

    # print("observation:", obs)
    states.append(obs)
    start_action = np.random.randint(0, 2)
    # if start_action == 1:
        # print("hit")
    # elif start_action == 0:
        # print("stick")
    obs, reward, done, _ = env.step(start_action)
    actions.append(start_action)
    # print("reward:", reward, "\n")
    ret += reward  # Note that gamma = 1. in this exercise

    while not done:
        # print("observation:", obs)
        states.append(obs)
        if obs[0] >= 12:
            action = int(pi[obs[0] - 12, obs[1] - 1, int(obs[2])])  # Get action from policy according to last (current) state
        else:
            action = 1

        obs, reward, done, _ = env.step(action)
        actions.append(action)
        # if action == 1:
            # print("hit")
        # elif action == 0:
            # print("stick")

        # print("reward:", reward, "\n")
        ret += reward  # Note that gamma = 1. in this exercise
    # print("final observation:", obs)
    return states, ret, actions


def visualize(V):
    plt.style.use('_mpl-gallery')

    xs = [[i for i in range(1, 11)] for _ in range(10)]
    ys = [[i for i in range(1, 11)] for _ in range(12, 22)]

    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_wireframe(xs, ys, V[:, :, 1])

    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player sum')
    ax.set_zlabel('Value')
    ax.set_title('No usable ace')

    plt.show()


def main():
    # single_run_20()
    # visualize(policy_evaluation())
    monte_carlo_es()


if __name__ == "__main__":
    main()
