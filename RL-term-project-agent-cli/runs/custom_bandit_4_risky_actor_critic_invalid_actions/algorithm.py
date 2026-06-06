import random
import csv
import math
from typing import List, Dict, Tuple, Optional

random.seed(42)

def transition(state: int, action: int) -> Tuple[int, int, bool]:
    # Clamp action to 0 or 1
    valid_action = 0 if action not in (0, 1) else action
    # Compute next state
    if valid_action == 0:
        next_state = max(0, state - 1)
    else:
        next_state = min(9, state + 1)
    done = next_state == 9
    reward = 10 if done else -1
    return next_state, reward, done

def main():
    gamma = 0.9
    alpha = 0.1
    actor_prefs = [0.0] * 10  # for states 0-9
    critic_values = [0.0] * 10

    # Train for 100 episodes
    episodes = 100
    report_data = []

    for episode in range(episodes):
        state = 0
        steps = 0
        total_reward = 0
        done = False

        while not done and steps < 35:
            # Compute action probabilities using softmax
            prob0 = 1 / (1 + math.exp(-actor_prefs[state]))
            prob1 = 1 - prob0
            action = random.choices([0, 1], weights=[prob0, prob1], k=1)[0]

            # Test invalid action with 0.1 probability
            if random.random() < 0.1:
                test_action = random.randint(0, 3)
                next_state, reward, done = transition(state, test_action)
            else:
                next_state, reward, done = transition(state, action)

            # Compute TD error
            td_error = reward + gamma * critic_values[next_state] - critic_values[state]

            # Update critic
            critic_values[state] += alpha * td_error

            # Update actor
            actor_prefs[state] += alpha * td_error * (1 if action == 0 else -1)

            state = next_state
            total_reward += reward
            steps += 1

        # Record episode data
        termination = "goal" if done else "max_steps"
        report_data.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "termination": termination
        })
        print(f"Episode {episode}: reward={total_reward}, steps={steps}, terminated at {termination}")

    # Generate CSV report
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'steps', 'termination'])
        for data in report_data:
            writer.writerow([data['episode'], data['total_reward'], data['steps'], data['termination']])

    # Textual policy and value summaries
    print("\nPolicy and value summaries:")
    for state in range(10):
        prob0 = 1 / (1 + math.exp(-actor_prefs[state]))
        prob1 = 1 - prob0
        print(f"State {state}: action 0 ({prob0:.2f}), action 1 ({prob1:.2f}) | Value: {critic_values[state]:.2f}")

    # Evaluate for 10 greedy episodes
    print("\nEvaluating with greedy policy...")
    greedy_episodes = 10
    greedy_report = []
    for _ in range(greedy_episodes):
        state = 0
        steps = 0
        total_reward = 0
        done = False
        while not done and steps < 35:
            # Greedy action selection
            action = 0 if actor_prefs[state] > 0 else 1
            next_state, reward, done = transition(state, action)
            state = next_state
            total_reward += reward
            steps += 1
        termination = "goal" if done else "max_steps"
        greedy_report.append({
            "episode": _,
            "total_reward": total_reward,
            "steps": steps,
            "termination": termination
        })
    print(f"Greedy evaluation complete. Average reward: {sum(item['total_reward'] for item in greedy_report) / greedy_episodes:.2f}")

if __name__ == '__main__':
    main()
