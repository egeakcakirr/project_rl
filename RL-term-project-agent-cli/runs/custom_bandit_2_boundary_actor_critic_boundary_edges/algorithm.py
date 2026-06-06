"""Algorithm for 1D grid Actor-Critic RL.

Implements a tabular Actor-Critic algorithm for a 1D grid environment with edge-case handling.
"""

from __future__ import annotations

import random
import math
import csv

random.seed(42)

def step(s: int, a: int) -> tuple[int, float, bool]:
    """Step in the 1D grid environment.
    
    Args:
        s: current state (0-9)
        a: action (0: left, 1: right)
    
    Returns:
        next_s: next state
        reward: reward for this step
        done: whether episode terminated
    """
    if s == 9:
        if a == 1:
            next_s = 9
            reward = 10.0
            done = True
        else:
            next_s = 8
            reward = -1.0
            done = False
    else:
        if a == 0:
            next_s = s - 1
        else:
            next_s = s + 1
        reward = -1.0
        done = (next_s == 9)
    # Ensure next_s is within 0-9
    if next_s < 0:
        next_s = 0
    elif next_s > 9:
        next_s = 9
    return next_s, reward, done

if __name__ == '__main__':
    # Initialize actor and critic
    gamma = 0.95
    alpha = 0.1

    # Initialize critic values (state values)
    critic_values: dict[int, float] = {s: 0.0 for s in range(10)}

    # Initialize actor parameters (theta for each state)
    actor_params: dict[int, list[float]] = {s: [0.0, 0.0] for s in range(10)}

    # Training loop
    num_episodes = 100
    episodes: list[dict] = []

    for episode in range(num_episodes):
        s = 0
        steps = 0
        total_reward = 0.0
        done = False
        while not done and steps < 35:
            # Compute action probabilities
            exp0 = math.exp(actor_params[s][0])
            exp1 = math.exp(actor_params[s][1])
            sum_exp = exp0 + exp1
            pi0 = exp0 / sum_exp
            pi1 = exp1 / sum_exp

            # Select action using softmax
            action = random.choices([0, 1], weights=[pi0, pi1], k=1)[0]

            # Take step
            next_s, reward, done = step(s, action)
            total_reward += reward
            steps += 1

            # Compute TD error
            if done:
                delta = reward - critic_values[s]
            else:
                delta = reward + gamma * critic_values[next_s] - critic_values[s]

            # Update critic
            critic_values[s] += alpha * delta

            # Update actor
            actor_params[s][0] += alpha * delta * (pi0 * pi1)
            actor_params[s][1] += alpha * delta * (pi1 * pi0)

        episodes.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'termination': 'goal' if done else 'max_steps'
        })

    # Evaluate for 10 greedy episodes
    greedy_episodes: list[dict] = []
    for _ in range(10):
        s = 0
        steps = 0
        total_reward = 0.0
        done = False
        while not done and steps < 35:
            # Greedy action selection
            exp0 = math.exp(actor_params[s][0])
            exp1 = math.exp(actor_params[s][1])
            sum_exp = exp0 + exp1
            pi0 = exp0 / sum_exp
            pi1 = exp1 / sum_exp

            action = 0 if pi0 > pi1 else 1

            next_s, reward, done = step(s, action)
            total_reward += reward
            steps += 1

        greedy_episodes.append({
            'episode': f'greedy_{_}',
            'total_reward': total_reward,
            'steps': steps,
            'termination': 'goal' if done else 'max_steps'
        })

    # Generate CSV report
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'steps', 'termination'])
        for ep in episodes:
            writer.writerow([ep['episode'], ep['total_reward'], ep['steps'], ep['termination']])

    # Print training results
    print("Training results:")
    for ep in episodes:
        print(f"Episode {ep['episode']}: Reward={ep['total_reward']:.2f}, Steps={ep['steps']}, Termination={ep['termination']}")

    # Print evaluation results
    print("\nEvaluation results (10 greedy episodes):")
    for ep in greedy_episodes:
        print(f"Greedy Episode {ep['episode']}: Reward={ep['total_reward']:.2f}, Steps={ep['steps']}, Termination={ep['termination']}")
