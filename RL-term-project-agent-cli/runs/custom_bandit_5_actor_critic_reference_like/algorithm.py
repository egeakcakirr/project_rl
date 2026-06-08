from typing import List, Dict, Optional, Any, Tuple
import random
import time
import csv
import os

def env_step(state: int, action: int) -> Tuple[int, int, bool]:
    """Take a step in the 1D grid environment."""
    if action == 0:  # left
        next_state = max(0, state - 1)
    else:  # right
        next_state = min(9, state + 1)
    reward = 10 if next_state == 9 else -1
    done = next_state == 9
    return next_state, reward, done

if __name__ == '__main__':
    # Configuration
    gamma = 0.95  # Discount factor
    alpha = 0.01  # Learning rate for critic
    max_steps = 35  # Maximum steps per episode
    episodes = 100  # Number of training episodes

    # Initialize actor and critic
    actor: Dict[int, int] = {i: 0 for i in range(10)}  # Always left initially
    critic: Dict[int, float] = {i: 0.0 for i in range(10)}  # Initial values

    training_results = []

    print("Starting training...")

    for episode in range(episodes):
        state = 0
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            # Choose action based on current state
            action = actor[state]
            next_state, reward, done = env_step(state, action)
            total_reward += reward
            steps += 1

            if done:
                break

            # Update critic using TD(0)
            td_error = reward + gamma * critic[next_state] - critic[state]
            critic[state] += alpha * td_error

            # Update actor: choose action that maximizes next state's critic value
            left_next = max(0, state - 1)
            right_next = min(9, state + 1)
            left_value = critic[left_next]
            right_value = critic[right_next]
            if left_value > right_value:
                actor[state] = 0
            else:
                actor[state] = 1

        training_results.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps
        })

    # Evaluation
    print("\nEvaluating policy...")
    state = 0
    total_reward = 0
    steps = 0
    for step in range(max_steps):
        action = actor[state]
        next_state, reward, done = env_step(state, action)
        total_reward += reward
        steps += 1
        if done:
            break
        state = next_state

    print(f"Evaluated: Total reward = {total_reward}, Steps = {steps}")

    # Generate CSV report
    csv_path = 'generation_report.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'total_reward', 'steps'])
        writer.writeheader()
        for result in training_results:
            writer.writerow(result)

    # Textual visualization
    print("\nPolicy and Value Function after training:")
    print("State -> Action (0: left, 1: right)")
    for state in range(10):
        print(f"{state}: {actor[state]}")

    print("\nValue Function (State -> Value):")
    for state in range(10):
        print(f"{state}: {critic[state]:.2f}")
