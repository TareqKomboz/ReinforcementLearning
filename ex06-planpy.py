import gym
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, parent=None, action=None):
        self.parent = parent  # parent of this node
        self.action = action  # action leading from parent to this node
        self.children = []
        self.sum_value = 0.  # sum of values observed for this node, use sum_value/visits for the mean
        self.visits = 0


def rollout(env, maxsteps=100):
    """ Random policy for rollouts """
    G = 0
    for _ in range(maxsteps):
        action = env.action_space.sample()
        _, reward, terminal, _ = env.step(action)
        G += reward
        if terminal:
            return G
    return G


def mcts_notree(env, root, maxiter=500):
    root.children = [Node(root, a) for a in range(env.action_space.n)]
    for _ in range(maxiter):
        state = copy.deepcopy(env)
        G = 0.

        # This is an example howto randomly choose a node and perform the action:
        node = random.choice(root.children)
        _, reward, terminal, _ = state.step(node.action)
        G += reward
        # This performs a rollout (Simulation):
        if not terminal:
            G += rollout(state)
        # This updates values for the current node:
        node.visits += 1
        node.sum_value += G


def epsilon_greedy(action_qs, epsilon):
    if np.random.rand() < 1 - epsilon:
        return np.asarray(action_qs).argmax()
    else:
        return np.random.randint(len(action_qs)) - 1


def mcts(env, root, maxiter=500):
    """ TODO: Use this function as a starting point for implementing Monte Carlo Tree Search
    """
    global all_treedepths

    if False:
        mcts_notree(env, root, maxiter)
        return

    # this is an example of how to add nodes to the root for all possible actions:
    root.children = [Node(root, a) for a in range(env.action_space.n)]
    treedepths = []

    for _ in range(maxiter):
        state = copy.deepcopy(env)
        G = 0.

        # TODO: traverse the tree using an epsilon greedy tree policy
        node = root
        terminal = False
        iter_treedepth = 0
        while len(node.children) > 0 and not terminal:
            values = [((c.sum_value/c.visits) if c.visits != 0 else 0) for c in node.children]  # calculate values for child actions
            node = node.children[epsilon_greedy(values, 0.4)]
            _, reward, terminal, _ = state.step(node.action)
            G += reward
            iter_treedepth += 1

        treedepths.append(iter_treedepth)

        # TODO: Expansion of tree
        if not terminal:
            node.children = [Node(node, a) for a in range(env.action_space.n)]

        # This performs a rollout (Simulation):
        if not terminal:
            G += rollout(state)

        # TODO: update all visited nodes in the tree
        while node.parent != None:
            node.visits += 1
            node.sum_value += G

            node = node.parent

    print(treedepths)
    all_treedepths.append(max(treedepths))
    print(max(treedepths))


all_treedepths = []


def main():
    global all_treedepths
    env = gym.make("Taxi-v3")
    env.seed(0)  # use seed to make results better comparable
    # run the algorithm 10 times:
    rewards = []
    episodes = 10
    for i in range(episodes):
        env.reset()
        terminal = False
        root = Node()  # Initialize empty tree
        sum_reward = 0.
        while not terminal:
            env.render()
            mcts(env, root)  # expand tree from root node using mcts
            values = [c.sum_value/c.visits for c in root.children]  # calculate values for child actions
            bestchild = root.children[np.argmax(values)]  # select the best child
            _, reward, terminal, _ = env.step(bestchild.action) # perform action for child
            root = bestchild  # use the best child as next root
            root.parent = None
            sum_reward += reward
        rewards.append(sum_reward)
        print("finished run " + str(i+1) + " with reward: " + str(sum_reward))
        plt.plot(range(len(all_treedepths)), all_treedepths, label=str(i))
        all_treedepths = []
    plt.legend()
    plt.show()
    print("mean reward: ", np.mean(rewards))
    plot(range(len(rewards)), rewards)


def plot(episodes, rewards):
    plt.plot(episodes, rewards)
    plt.show()


if __name__ == "__main__":
    main()
