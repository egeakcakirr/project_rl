"""Algorithm target file for mecha-agent-cli.

This file implements a complete Actor-Critic reinforcement learning application for a 1D grid world.
"""

from __future__ import annotations

import random
import math
import csv
from typing import List, Dict, Tuple, Optional

class GridWorldEnv:
    def __init__(self, max_steps: int = 40):
        self.max_steps = max_steps
        self.current_state = 0
        self.steps = 0

    def reset(self) -> int:
        self.current_state = 0
        self.steps = 0
        return self.current_state

    def step(self, action: int) -> Tuple[int, int, bool]:
        self.steps += 1
        if action == 0:  # left
            next_state = max(0, self.current_state - 1)
        else:  # right
            next_state = min(9, self.current_state + 1)
        reward = 10 if next_state == 9 else -1
        done = (next_state == 9) or (self.steps >= self.max_steps)
        self.current_state = next_state
        return next_state, reward, done

if __name__ == '__main__':
    random.seed(42)
    gamma = 0.95
    alpha_critic = 0.1
    alpha_actor = 0.01

    actor = [[0.0, 0.0] for _ in range(10)]
    critic = [0.0] * 10

    training_episodes = []
    for episode in range(150):
        env = GridWorldEnv(max_steps=40)
        current_state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        next_state = current_state
        while not done:
            # Compute action probabilities
            exp_prefs = [math.exp(actor[current_state][a]) for a in range(2)]
            probs = [p / sum(exp_prefs) for p in exp_prefs]
            action = random.choices([0, 1], weights=probs)[0]

            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

            # Compute TD error
            td_error = reward + gamma * critic[next_state] - critic[current_state]

            # Update critic
            critic[current_state] += alpha_critic * td_error

            # Update actor
            for a in range(2):
                exp_val = math.exp(actor[current_state][a])
                total_exp = sum([math.exp(actor[current_state][a]) for a in range(2)])
                p = exp_val / total_exp
                actor[current_state][a] += alpha_actor * td_error * p

            current_state = next_state

        termination_reason = "goal" if next_state == 9 else "timeout"
        training_episodes.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'termination': termination_reason
        })
        print(f"Episode {episode}: Reward={total_reward}, Steps={steps}, Termination={termination_reason}")

    # Evaluation loop
    evaluation_episodes = []
    for _ in range(10):
        env = GridWorldEnv(max_steps=40)
        current_state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done:
            # Greedy action selection
            action = 0 if actor[current_state][0] > actor[current_state][1] else 1
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
        evaluation_episodes.append({
            'total_reward': total_reward,
            'steps': steps
        })

    # Print evaluation summary
    print("\nEvaluation Results:")
    avg_reward = sum(e['total_reward'] for e in evaluation_episodes) / len(evaluation_episodes)
    avg_steps = sum(e['steps'] for e in evaluation_episodes) / len(evaluation_episodes)
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.2f}")

    # Print textual policy visualization
    print("\nPolicy Visualization:")
    for state in range(10):
        action_probs = [math.exp(actor[state][a]) for a in range(2)]
        total = sum(action_probs)
        probs = [p / total for p in action_probs]
        print(f"State {state}: Action 0 ({probs[0]:.2f}), Action 1 ({probs[1]:.2f})")

    # Print value function summary
    print("\nValue Function Summary:")
    for state in range(10):
        print(f"State {state}: Value = {critic[state]:.2f}")

    # Print learning progress summary
    print("\nLearning Progress Summary:")
    initial_avg_reward = sum(training_episodes[:10]['total_reward']) / 10
    final_avg_reward = sum(e['total_reward'] for e in training_episodes) / 150
    print(f"Initial average reward: {initial_avg_reward:.2f}")
    print(f"Final average reward: {final_avg_reward:.2f}")

    # Write CSV report
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'total_reward', 'steps', 'termination'])
        writer.writeheader()
        for entry in training_episodes:
            writer.writerow(entry)
