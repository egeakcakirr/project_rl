"""Algorithm for a stress-tested Actor-Critic RL application in a 1D grid world.

This implementation trains an actor-critic model on a 1D grid world with states 0..14.
The agent learns to navigate from state 0 to state 14 with a reward of +15 at the goal.
The training runs for 500 episodes, and evaluation for 20 episodes.
The code includes a CSV report, textual visualization, and strict max_steps guards.
"""

import random
import csv
import math
from typing import List, Dict, Tuple, Optional

def step(state: int, action: int) -> Tuple[int, int]:
    """Take an action in the 1D grid world environment.

    Args:
        state: Current state (0-14)
        action: Action (0: left, 1: right)

    Returns:
        next_state: Next state (0-14)
        reward: Reward (-1 or 15)
    """
    if action == 0:
        next_state = max(0, state - 1)
    else:
        next_state = min(14, state + 1)
    reward = 15 if next_state == 14 else -1
    return next_state, reward

def main():
    random.seed(42)
    gamma = 0.99
    alpha = 0.1

    # Initialize critic and actor
    critic: List[float] = [0.0] * 15
    actor_policy: List[List[float]] = [[0.0, 0.0] for _ in range(15)]

    # Training loop
    training_episodes = 500
    report_data = []

    print("Starting training...")
    for episode in range(1, training_episodes + 1):
        state = 0
        steps = 0
        total_reward = 0
        termination = False

        while steps < 70 and state != 14:
            # Compute action probabilities using softmax
            exp_vals = [
                math.exp(actor_policy[state][0]),
                math.exp(actor_policy[state][1])
            ]
            probs = [exp_val / sum(exp_vals) for exp_val in exp_vals]
            action = random.choices([0, 1], weights=probs)[0]

            next_state, reward = step(state, action)
            steps += 1
            total_reward += reward

            # Compute TD error
            td_error = reward + gamma * critic[next_state] - critic[state]

            # Update critic
            critic[state] += alpha * td_error

            # Update actor
            actor_policy[state][action] += alpha * td_error

            state = next_state

        # Episode ends
        termination = (state == 14)
        report_data.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'termination': termination
        })
        print(f"Episode {episode}: Total reward = {total_reward}, Steps = {steps}, Termination = {termination}")

    # Evaluation
    print("\nStarting evaluation...")
    evaluation_results = []
    for episode in range(1, 21):
        state = 0
        steps = 0
        total_reward = 0
        termination = False

        while steps < 70 and state != 14:
            # Compute next states for both actions
            left_state = max(0, state - 1)
            right_state = min(14, state + 1)
            
            # Choose action with higher critic value for next state
            if critic[left_state] > critic[right_state]:
                action = 0
            else:
                action = 1
                
            next_state, reward = step(state, action)
            steps += 1
            total_reward += reward
            state = next_state

        termination = (state == 14)
        evaluation_results.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'termination': termination
        })

    # Generate CSV report
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'total_reward', 'steps', 'termination'])
        writer.writeheader()
        writer.writerows(report_data)

    # Textual visualization
    print("\nPolicy and value visualization:")
    for state in range(15):
        if critic[state] > 0:
            policy_action = 'R'
        else:
            policy_action = 'L'
        print(f"State {state}: Policy={policy_action}, Value={critic[state]:.2f}")

    # Print evaluation summary
    print("\nEvaluation summary:")
    for result in evaluation_results:
        print(f"Episode {result['episode']}: Reward={result['total_reward']}, Steps={result['steps']}, Termination={result['termination']}")

if __name__ == '__main__':
    main()
