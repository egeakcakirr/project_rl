from typing import List, Dict, Tuple, Optional, Any
import math
import random
import csv
import os
import time

def safe_transition(s: int, a: int) -> int:
    """Safely transition to next state in 1D grid."""
    next_s = s + (1 if a == 1 else -1)
    return max(0, min(9, next_s))

def get_reward(s: int) -> int:
    """Reward for reaching goal state."""
    return 10 if s == 9 else -1

def select_action(state: int, actor_logits: List[List[float]]) -> int:
    """Select action using softmax for training."""
    exp_vals = [math.exp(x) for x in actor_logits[state]]
    total = sum(exp_vals)
    probs = [exp_val / total for exp_val in exp_vals]
    return 0 if probs[0] > probs[1] else 1

def select_greedy(state: int, actor_logits: List[List[float]]) -> int:
    """Select action using greedy policy (max probability)."""
    exp_vals = [math.exp(x) for x in actor_logits[state]]
    total = sum(exp_vals)
    probs = [exp_val / total for exp_val in exp_vals]
    return 0 if probs[0] > probs[1] else 1

if __name__ == '__main__':
    random.seed(42)
    gamma = 0.95
    alpha = 0.1

    # Initialize critic and actor
    V = [0.0] * 10
    actor_logits = [[0.0, 0.0] for _ in range(10)]

    training_episodes = 100
    evaluation_episodes = 10

    all_episodes = []

    # Train for 100 episodes
    print("Training for 100 episodes...")
    for episode in range(training_episodes):
        state = 0
        steps = 0
        total_reward = 0
        done = False

        while steps < 35 and not done:
            action = select_action(state, actor_logits)
            next_state = safe_transition(state, action)
            reward = get_reward(next_state)
            total_reward += reward

            # Update critic
            if next_state == 9:
                delta = reward - V[state]
            else:
                delta = reward + gamma * V[next_state] - V[state]
            V[state] += alpha * delta

            # Update actor
            actor_logits[state][action] += alpha * delta

            state = next_state
            steps += 1

            if next_state == 9:
                done = True

        termination_reason = "goal" if next_state == 9 else "max_steps"
        all_episodes.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'termination': termination_reason
        })
        print(f"Episode {episode}: Reward={total_reward}, Steps={steps}, Termination={termination_reason}")

    # Evaluate for 10 episodes
    print("\nEvaluating for 10 episodes...")
    for episode in range(evaluation_episodes):
        state = 0
        steps = 0
        total_reward = 0
        done = False

        while steps < 35 and not done:
            action = select_greedy(state, actor_logits)
            next_state = safe_transition(state, action)
            reward = get_reward(next_state)
            total_reward += reward

            state = next_state
            steps += 1

            if next_state == 9:
                done = True

        termination_reason = "goal" if next_state == 9 else "max_steps"
        all_episodes.append({
            'episode': training_episodes + episode,
            'total_reward': total_reward,
            'steps': steps,
            'termination': termination_reason
        })
        print(f"Episode {training_episodes + episode}: Reward={total_reward}, Steps={steps}, Termination={termination_reason}")

    # Generate CSV report
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'steps', 'termination'])
        for ep in all_episodes:
            writer.writerow([
                ep['episode'],
                ep['total_reward'],
                ep['steps'],
                ep['termination']
            ])

    # Print textual policy visualization
    print("\nPolicy visualization:")
    for s in range(10):
        exp0 = math.exp(actor_logits[s][0])
        exp1 = math.exp(actor_logits[s][1])
        total = exp0 + exp1
        p0 = exp0 / total
        p1 = exp1 / total
        print(f"State {s}: Action 0: {p0:.2f}, Action 1: {p1:.2f}")
