"""Compact Actor-Critic RL application for 1D grid world."""

from __future__ import annotations

import math
import random
import csv
from typing import Dict, List, Optional, Tuple

random.seed(42)

gamma = 0.9
alpha = 0.1
max_steps = 25
num_episodes = 30
num_evaluation_episodes = 5

actor: Dict[int, List[float]] = {s: [0.0, 0.0] for s in range(10)}
critic: Dict[int, float] = {s: 0.0 for s in range(10)}

if __name__ == '__main__':
    episodes_data = []
    for episode in range(num_episodes):
        state = 0
        steps = 0
        total_reward = 0
        while steps < max_steps:
            # Compute action probabilities using softmax
            exp_prefs = [math.exp(actor[state][0]), math.exp(actor[state][1])]
            total_prob = sum(exp_prefs)
            p_left = exp_prefs[0] / total_prob
            p_right = exp_prefs[1] / total_prob
            action = 0 if random.random() < p_left else 1

            # Clamp transition
            if action == 0:
                next_state = max(0, state - 1)
            else:
                next_state = min(9, state + 1)

            # Compute reward
            reward = 10 if next_state == 9 else -1

            # Check for goal termination
            if next_state == 9:
                break

            # Compute TD error
            delta = reward + gamma * critic[next_state] - critic[state]

            # Update critic
            critic[state] += alpha * delta

            # Update actor for the taken action
            actor[state][action] += alpha * delta

            steps += 1
            total_reward += reward

        # Record episode data
        termination = 'goal' if next_state == 9 else 'max_steps'
        episodes_data.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'termination': termination
        })

    # Write report to CSV
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'total_reward', 'steps', 'termination'])
        writer.writeheader()
        for data in episodes_data:
            writer.writerow(data)

    # Evaluate policy
    print("Training completed. Evaluating policy...")
    evaluation_data = []
    for _ in range(num_evaluation_episodes):
        state = 0
        steps = 0
        total_reward = 0
        while steps < max_steps:
            # Greedy action selection
            action = 0 if actor[state][0] > actor[state][1] else 1

            # Clamp transition
            if action == 0:
                next_state = max(0, state - 1)
            else:
                next_state = min(9, state + 1)

            # Compute reward
            reward = 10 if next_state == 9 else -1

            # Check for goal termination
            if next_state == 9:
                break

            steps += 1
            total_reward += reward

        # Record evaluation data
        termination = 'goal' if next_state == 9 else 'max_steps'
        evaluation_data.append({
            'episode': f'eval_{_}',
            'total_reward': total_reward,
            'steps': steps,
            'termination': termination
        })

    # Print evaluation results
    print("\nEvaluation results:")
    for i, data in enumerate(evaluation_data):
        print(f"Episode {i+1}: Reward={data['total_reward']}, Steps={data['steps']}, Termination={data['termination']}")
