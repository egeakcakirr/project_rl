"""Algorithm for 1D grid Actor-Critic training with negative rewards.

This file implements a tabular Actor-Critic agent for a 1D grid environment.
"""

from __future__ import annotations

import random
import csv
import math

random.seed(42)

class GridEnv:
    def __init__(self, max_steps: int = 35):
        self.max_steps = max_steps
        self.current_state = 0
        self.steps = 0

    def step(self, action: int) -> tuple[int, int, bool]:
        if self.steps >= self.max_steps:
            return self.current_state, 0, True
        
        if action == 0:
            next_state = max(0, self.current_state - 1)
        else:
            next_state = min(9, self.current_state + 1)
        
        self.steps += 1
        reward = -2
        if next_state == 9:
            reward = 10
            done = True
        else:
            done = False
        self.current_state = next_state
        return next_state, reward, done

def main():
    """Train and evaluate Actor-Critic agent for 1D grid environment."""
    critic_values = [[0.0] * 2 for _ in range(10)]
    alpha = 0.1
    gamma = 1.0
    
    report_data = []
    for episode in range(120):
        env = GridEnv()
        total_reward = 0
        steps = 0
        done = False
        current_state = 0
        while not done and steps < 35:
            # Compute action probabilities using softmax
            exp_vals = [math.exp(critic_values[current_state][a]) for a in range(2)]
            probs = [exp_val / sum(exp_vals) for exp_val in exp_vals]
            action = random.choices([0, 1], weights=probs)[0]
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            # TD error update
            if done:
                next_q = 0
            else:
                next_q = max(critic_values[next_state])
            td_error = reward + gamma * next_q - critic_values[current_state][action]
            critic_values[current_state][action] += alpha * td_error
        
        report_data.append({
            'episode': episode + 1,
            'total_reward': total_reward,
            'steps': steps,
            'termination': 'goal' if done else 'max_steps'
        })
    
    # Write report to CSV
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'total_reward', 'steps', 'termination'])
        writer.writeheader()
        writer.writerows(report_data)
    
    # Evaluate greedy policy
    print("\nEvaluating greedy policy (10 episodes):")
    greedy_total_reward = 0
    for _ in range(10):
        env = GridEnv()
        total_reward = 0
        steps = 0
        current_state = 0
        done = False
        while not done and steps < 35:
            # Greedy action selection
            action = 0 if critic_values[current_state][0] > critic_values[current_state][1] else 1
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
        greedy_total_reward += total_reward
    
    avg_reward = greedy_total_reward / 10
    print(f"Average reward: {avg_reward:.2f}")
    
    # Textual policy and value visualization
    print("\nPolicy and Value Visualization:")
    for state in range(10):
        exp_vals = [math.exp(critic_values[state][a]) for a in range(2)]
        probs = [exp_val / sum(exp_vals) for exp_val in exp_vals]
        print(f"State {state}: Left ({probs[0]:.2f}), Right ({probs[1]:.2f})")
        print(f"Value: {critic_values[state][0]:.2f} (Left), {critic_values[state][1]:.2f} (Right)")

if __name__ == '__main__':
    main()
