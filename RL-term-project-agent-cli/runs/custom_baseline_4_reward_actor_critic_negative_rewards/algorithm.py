import random
import csv
import math
from typing import List, Dict, Tuple, Optional

random.seed(42)

GOAL_STATE = 9
MAX_STEPS = 35
REWARD_GOAL = 10
REWARD_STEP = -2
GAMMA = 0.9
ALPHA = 0.1

def main() -> None:
    """Train and evaluate a tabular actor-critic agent on a 1D grid environment.

    The agent learns to navigate from state 0 to state 9 with negative rewards
    for each step. After training, it evaluates the agent's performance with
    a greedy policy.
    """
    actor_preferences: List[List[float]] = [[0.0, 0.0] for _ in range(10)]
    critic_values: List[float] = [0.0] * 10

    report_data: List[Dict[str, object]] = []

    for episode in range(120):
        current_state = 0
        steps = 0
        total_reward = 0
        terminated = False

        while steps < MAX_STEPS:
            # Compute action probabilities via softmax
            q_left = actor_preferences[current_state][0]
            q_right = actor_preferences[current_state][1]
            probs = [math.exp(q_left), math.exp(q_right)]
            probs = [p / sum(probs) for p in probs]

            # Select action
            action = random.choices([0, 1], weights=probs, k=1)[0]

            # Determine next state (clamped)
            if action == 0:  # left
                next_state = max(0, current_state - 1)
            else:  # right
                next_state = min(9, current_state + 1)

            # Compute reward
            if next_state == GOAL_STATE:
                reward = REWARD_GOAL
                terminated = True
            else:
                reward = REWARD_STEP

            # Update total reward and steps
            total_reward += reward
            steps += 1

            # Update critic values
            td_error = reward + GAMMA * critic_values[next_state] - critic_values[current_state]
            critic_values[current_state] += ALPHA * td_error

            # Update actor preferences for the action taken
            if action == 0:
                actor_preferences[current_state][0] += ALPHA * td_error
            else:
                actor_preferences[current_state][1] += ALPHA * td_error

            current_state = next_state

            if terminated:
                break

        # Record episode data
        termination_reason = "goal" if terminated else "max_steps"
        report_data.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "steps": steps,
            "termination": termination_reason
        })

    # Save report to CSV
    with open("generation_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "total_reward", "steps", "termination"])
        writer.writeheader()
        writer.writerows(report_data)

    # Evaluate with greedy policy
    print("\nEvaluating with greedy policy...")
    greedy_episodes = 10
    total_evaluation_reward = 0.0
    for _ in range(greedy_episodes):
        current_state = 0
        steps = 0
        episode_reward = 0
        terminated = False

        while steps < MAX_STEPS:
            # Greedy action selection
            if actor_preferences[current_state][0] > actor_preferences[current_state][1]:
                action = 0
            else:
                action = 1

            # Determine next state (clamped)
            if action == 0:  # left
                next_state = max(0, current_state - 1)
            else:  # right
                next_state = min(9, current_state + 1)

            # Compute reward
            if next_state == GOAL_STATE:
                reward = REWARD_GOAL
                terminated = True
            else:
                reward = REWARD_STEP

            episode_reward += reward
            steps += 1

            current_state = next_state

            if terminated:
                break

        total_evaluation_reward += episode_reward

    avg_evaluation_reward = total_evaluation_reward / greedy_episodes
    print(f"Average evaluation reward: {avg_evaluation_reward:.2f}")

    # Textual visualization
    print("\nTextual policy and value visualization:")
    print("State | Policy (left/right) | Critic Value")
    for state in range(10):
        left_prob = math.exp(actor_preferences[state][0]) / (math.exp(actor_preferences[state][0]) + math.exp(actor_preferences[state][1]))
        right_prob = 1 - left_prob
        print(f"{state} | left: {left_prob:.2f}, right: {right_prob:.2f} | {critic_values[state]:.2f}")

if __name__ == "__main__":
    main()
