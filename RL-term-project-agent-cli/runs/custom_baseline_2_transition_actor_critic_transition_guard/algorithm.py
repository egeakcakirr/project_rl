"""Algorithm for 1D grid world Actor-Critic RL agent.

This implementation trains an actor-critic agent on a 1D grid world with states 0-9.
The agent learns to reach state 9 (goal) with a reward of +10, avoiding -1 per step.
The agent uses tabular actor and critic values, softmax policy, and TD error updates.
"""

import random
import csv
import math
from typing import Dict, List, Tuple, Optional

def transition(state: int, action: int) -> Tuple[int, int]:
    """Compute next state and reward for given state and action.

    Args:
        state: Current state (int)
        action: Action (0: left, 1: right)

    Returns:
        next_state: Next state (int)
        reward: Reward (int)
    """
    if action == 0:
        next_state = max(0, state - 1)
    else:
        next_state = min(9, state + 1)
    reward = 10 if next_state == 9 else -1
    return next_state, reward

def main():
    random.seed(42)
    gamma = 0.9
    alpha = 0.1

    # Initialize actor and critic
    actor: Dict[int, List[float]] = {s: [0.0, 0.0] for s in range(10)}
    critic: Dict[int, float] = {s: 0.0 for s in range(10)}

    # Training loop: 100 episodes
    training_episodes = 100
    evaluation_episodes = 10

    # CSV report
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'steps', 'termination'])

    # Train for 100 episodes
    for episode in range(1, training_episodes + 1):
        current_state = 0
        steps = 0
        total_reward = 0
        terminated = False
        while steps < 35:
            # Select action using current policy
            probs = [math.exp(actor[current_state][a]) for a in [0, 1]]
            probs = [p / sum(probs) for p in probs]
            action = 0 if probs[0] > probs[1] else 1

            next_state, reward = transition(current_state, action)
            total_reward += reward

            # Update critic
            next_critic = critic[next_state] if next_state != 9 else 0.0
            td_error = reward + gamma * next_critic - critic[current_state]
            critic[current_state] += alpha * td_error

            # Update actor
            for a in [0, 1]:
                actor[current_state][a] += alpha * td_error * probs[a]

            current_state = next_state
            steps += 1

            if next_state == 9:
                terminated = True
                break

        termination = 'goal' if next_state == 9 else 'max_steps'
        with open('generation_report.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, steps, termination])

    # Evaluate for 10 episodes
    print("\nTraining complete. Evaluating...")
    for episode in range(1, evaluation_episodes + 1):
        current_state = 0
        steps = 0
        total_reward = 0
        while steps < 35:
            # Greedy action selection
            probs = [math.exp(actor[current_state][a]) for a in [0, 1]]
            probs = [p / sum(probs) for p in probs]
            action = 0 if probs[0] > probs[1] else 1

            next_state, reward = transition(current_state, action)
            total_reward += reward

            current_state = next_state
            steps += 1
            if next_state == 9:
                break

        print(f"Episode {episode}: Reward={total_reward}, Steps={steps}, Termination={next_state}")

    # Textual policy visualization
    print("\nPolicy Visualization:")
    for state in range(10):
        probs = [math.exp(actor[state][a]) for a in [0, 1]]
        probs = [p / sum(probs) for p in probs]
        action = 0 if probs[0] > probs[1] else 1
        print(f"State {state}: Action {action} (prob: {probs[0]:.2f} vs {probs[1]:.2f})")

if __name__ == '__main__':
    main()
