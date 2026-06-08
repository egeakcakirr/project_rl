from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import random
import csv
import sys
import math

class GridWorldEnv:
    """1D grid world environment for RL training with states 0-9."""
    def __init__(self, max_steps: int = 25) -> None:
        self.max_steps = max_steps
        self.goal = 9

    def step(self, state: int, action: int) -> Tuple[int, int, bool]:
        """Simulate one step in the environment.
        
        Args:
            state: Current state (0-9)
            action: Action (0=left, 1=right)
        
        Returns:
            next_state: Next state (0-9)
            reward: Reward for this step
            done: Whether episode ended (goal reached or max steps)
        """
        if action == 0:
            next_state = max(0, state - 1)
        else:
            next_state = min(self.goal, state + 1)
        
        reward = 10 if next_state == self.goal else -1
        done = next_state == self.goal
        return next_state, reward, done

if __name__ == '__main__':
    random.seed(42)
    env = GridWorldEnv()
    num_states = 10
    num_actions = 2
    Q = [[0.0] * num_actions for _ in range(num_states)]
    V = [0.0] * num_states

    gamma = 0.9
    alpha = 0.1
    temp = 1.0

    episodes = 30
    episode_rewards = []
    episode_steps = []
    episode_terminations = []

    for episode in range(episodes):
        state = 0
        steps = 0
        total_reward = 0
        done = False
        while not done and steps < 25:
            # Compute softmax probabilities
            action_probs = [math.exp(Q[state][a] / temp) for a in range(num_actions)]
            action_probs = [p / sum(action_probs) for p in action_probs]
            action = random.choices([0, 1], weights=action_probs)[0]

            next_state, reward, done = env.step(state, action)
            total_reward += reward
            steps += 1

            # Compute TD error
            td_error = reward + gamma * V[next_state] - V[state]

            # Update critic (V)
            V[state] += alpha * td_error

            # Update actor (Q)
            Q[state][action] += alpha * td_error

            state = next_state

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_terminations.append("goal" if done else "max_steps")

        print(f"Episode {episode + 1}: Reward={total_reward}, Steps={steps}")

    # Write results to CSV
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'steps', 'termination'])
        for i in range(episodes):
            writer.writerow([i, episode_rewards[i], episode_steps[i], episode_terminations[i]])

    # Evaluate policy with 5 greedy episodes
    print("\nEvaluation results (5 greedy episodes):")
    for _ in range(5):
        state = 0
        steps = 0
        total_reward = 0
        done = False
        while not done and steps < 25:
            action = max(range(num_actions), key=lambda a: Q[state][a])
            next_state, reward, done = env.step(state, action)
            total_reward += reward
            steps += 1
            state = next_state
        print(f"Reward={total_reward}, Steps={steps}")

    # Visualize policy
    print("\nLearned policy (state -> action):")
    for state in range(num_states):
        action = max(range(num_actions), key=lambda a: Q[state][a])
        print(f"State {state}: Action {action}")
