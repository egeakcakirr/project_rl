import random
import csv
import sys
import math
from typing import Dict, List, Optional, Tuple

class Environment:
    def __init__(self, max_steps: int = 45):
        self.max_steps: int = max_steps
        self.current_state: int = 0
        self.steps: int = 0

    def reset(self) -> int:
        self.current_state = 0
        self.steps = 0
        return self.current_state

    def step(self, action: int) -> Tuple[int, int, bool]:
        self.steps += 1
        if action == 0:
            next_state = max(0, self.current_state - 1)
        else:
            next_state = min(9, self.current_state + 1)
        
        if next_state == 9:
            reward = 20
            done = True
        else:
            reward = -1
            done = (self.steps >= self.max_steps)
        
        self.current_state = next_state
        return next_state, reward, done

def main():
    """Train and evaluate the Actor-Critic agent for 150 episodes and generate report."""
    random.seed(42)
    gamma = 0.9
    actor_lr = 0.1
    critic_lr = 0.1

    actor_policy: Dict[int, List[float]] = {i: [0.0, 0.0] for i in range(10)}
    critic_values: Dict[int, float] = {i: 0.0 for i in range(10)}

    csv_filename = "generation_report.csv"
    episodes = []

    for episode in range(1, 151):
        env = Environment()
        current_state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 45:
            prefs = actor_policy[current_state]
            exp_prefs = [math.exp(p) for p in prefs]
            total_exp = sum(exp_prefs)
            probs = [exp_pref / total_exp for exp_pref in exp_prefs]
            action = 0 if random.random() < probs[0] else 1

            next_state, reward, done = env.step(action)

            if done:
                next_value = 0.0
            else:
                next_value = critic_values[next_state]

            td_error = reward + gamma * next_value - critic_values[current_state]
            critic_values[current_state] += critic_lr * td_error

            if action == 0:
                actor_policy[current_state][0] += actor_lr * td_error
            else:
                actor_policy[current_state][1] += actor_lr * td_error

            total_reward += reward
            steps += 1
            current_state = next_state

        termination_reason = "goal" if next_state == 9 else "max_steps"
        episodes.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "termination": termination_reason
        })

        print(f"Episode {episode}: Reward={total_reward}, Steps={steps}, Terminated due to {termination_reason}")

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "total_reward", "steps", "termination"])
        writer.writeheader()
        writer.writerows(episodes)

    print("\nLearning summary:")
    print(f"Trained for 150 episodes with actor learning rate={actor_lr}, critic learning rate={critic_lr}, gamma={gamma}")
    print(f"Final policy preferences (state 0): left={actor_policy[0][0]:.2f}, right={actor_policy[0][1]:.2f}")

if __name__ == '__main__':
    main()
