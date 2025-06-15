#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Define the bandit problem
class Bandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.means = np.random.normal(0, 1, arms)
        self.best_action = np.argmax(self.means)

    def pull(self, action):
        return np.random.normal(self.means[action], 1)

# Define the agent
class Agent:
    def __init__(self, arms=10, epsilon=0.1):
        self.arms = arms
        self.epsilon = epsilon
        self.Q = np.zeros(arms)
        self.N = np.zeros(arms)

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.arms)
        else:
            return np.argmax(self.Q)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

# Run the experiment
def run_experiment(bandit, agent, trials=1000):
    rewards = np.zeros(trials)
    best_action_counts = np.zeros(trials)
    for t in range(1,trials):
        action = agent.choose_action()
        reward = bandit.pull(action)
        agent.update(action, reward)
        rewards[t] = rewards[t-1] + 1/t*(reward-rewards[t-1])
        best_action_counts[t] = (action == bandit.best_action) + best_action_counts[t-1]
    return rewards, best_action_counts/np.arange(trials)

# Plot the results
def plot_results(rewards, best_action_counts, epsilon):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label=f'Epsilon: {epsilon}')
    plt.xlabel('Trials')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(best_action_counts, label=f'Epsilon: {epsilon}')
    plt.xlabel('Trials')
    plt.ylabel('Best Action Count')
    plt.legend()
    plt.show()

# Main function
if __name__ == "__main__":
    bandit = Bandit()
    agent = Agent(epsilon=0.2)
    rewards, best_action_counts = run_experiment(bandit, agent)
    plot_results(rewards, best_action_counts, agent.epsilon)

