import random
import csv
import math
from typing import List, Dict, Tuple, Optional

def step(state: int, action: int) -> Tuple[int, float, bool]:
    """Step the environment from current state with action.
    
    Args:
        state: Current state (int)
        action: Action (0 = left, 1 = right)
    
    Returns:
        next_state: Next state (int)
        reward: Reward (float)
        done: Episode termination flag (bool)
    """
    next_state = state + (1 if action == 1 else -1)
    next_state = max(0, min(9, next_state))
    if next_state == 9:
        return next_state, 20.0, True
    else:
        return next_state, -1.0, False

if __name__ == '__main__':
    random.seed(42)
    actor = [[0.0, 0.0] for _ in range(10)]
    critic = [0.0] * 10
    actor_lr = 0.01
    critic_lr = 0.01
    gamma = 0.95
    training_episodes = 150
    evaluation_episodes = 10

    # Training episodes
    training_data = []
    for episode in range(training_episodes):
        state = 0
        steps = 0
        total_reward = 0.0
        done = False
        while not done and steps < 45:
            # Compute action probabilities
            exp0 = math.exp(actor[state][0])
            exp1 = math.exp(actor[state][1])
            total = exp0 + exp1
            p0 = exp0 / total
            p1 = exp1 / total

            action = random.choices([0, 1], weights=[p0, p1])[0]

            next_state, reward, done_flag = step(state, action)
            total_reward += reward

            # Compute TD error
            delta = reward + gamma * critic[next_state] - critic[state]

            # Update critic
            critic[state] += critic_lr * delta

            # Update actor
            if action == 0:
                actor[state][0] += actor_lr * delta * p0
            else:
                actor[state][1] += actor_lr * delta * p1

            state = next_state
            steps += 1
            done = done_flag

        # Record episode data
        termination = 'goal' if done else 'max_steps'
        training_data.append({
            'episode': episode + 1,
            'total_reward': total_reward,
            'steps': steps,
            'termination': termination
        })
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}, Termination: {termination}")

    # Evaluation episodes
    evaluation_data = []
    for i in range(evaluation_episodes):
        state = 0
        steps = 0
        total_reward = 0.0
        done = False
        while not done and steps < 45:
            # Greedy action selection
            exp0 = math.exp(actor[state][0])
            exp1 = math.exp(actor[state][1])
            total = exp0 + exp1
            p0 = exp0 / total
            p1 = exp1 / total
            action = 0 if p0 > p1 else 1

            next_state, reward, done_flag = step(state, action)
            total_reward += reward
            state = next_state
            steps += 1
            done = done_flag

        termination = 'goal' if done else 'max_steps'
        evaluation_data.append({
            'episode': i + 1 + training_episodes,
            'total_reward': total_reward,
            'steps': steps,
            'termination': termination
        })

    # Generate CSV report
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'total_reward', 'steps', 'termination'])
        writer.writeheader()
        writer.writerows(training_data + evaluation_data)

    # Textual learning summary
    print("\nLearning summary:")
    print(f"Trained for {training_episodes} episodes with {actor_lr:.4f} actor learning rate and {critic_lr:.4f} critic learning rate.")
    print(f"Evaluated {evaluation_episodes} greedy episodes.")
    print(f"Average reward per training episode: {sum(data['total_reward'] for data in training_data) / training_episodes:.2f}")
    print(f"Average steps per training episode: {sum(data['steps'] for data in training_data) / training_episodes:.2f}")
