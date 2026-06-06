from typing import Dict, List, Tuple, Optional, Any
import random
import math
import csv

class GridWorldEnv:
    def __init__(self, max_steps: int = 40):
        self.state = 0
        self.steps = 0
        self.max_steps = max_steps

    def reset(self) -> int:
        self.state = 0
        self.steps = 0
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        self.steps += 1
        if self.steps > self.max_steps:
            done = True
            reward = -1
        else:
            if action == 0:
                next_state = max(0, self.state - 1)
            else:
                next_state = min(9, self.state + 1)
            if next_state == 9:
                done = True
                reward = 10
            else:
                done = False
                reward = -1
            self.state = next_state
        return next_state, reward, done

def select_action(state: int, actor: Dict[int, List[float]], temperature: float = 1.0) -> int:
    scores = actor[state]
    exp_scores = [math.exp(score) for score in scores]
    total = sum(exp_scores)
    probs = [exp_score / total for exp_score in exp_scores]
    return random.choices([0, 1], weights=probs, k=1)[0]

if __name__ == '__main__':
    gamma = 0.9
    alpha_critic = 0.1
    alpha_actor = 0.01
    random.seed(42)

    actor = {s: [0.0, 0.0] for s in range(10)}
    critic = {s: 0.0 for s in range(10)}

    episodes = []
    for episode in range(150):
        env = GridWorldEnv()
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done and steps < 40:
            action = select_action(state, actor)
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            delta = reward + gamma * critic[next_state] - critic[state]
            critic[state] += alpha_critic * delta
            scores = actor[state]
            exp_scores = [math.exp(score) for score in scores]
            prob = exp_scores[action] / sum(exp_scores)
            actor[state][action] += alpha_actor * delta * prob
        termination = "goal" if next_state == 9 else "timeout"
        episodes.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "steps": steps,
            "termination": termination
        })

    # Evaluation
    env_eval = GridWorldEnv()
    total_rewards = []
    steps_list = []
    for _ in range(10):
        state = env_eval.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done:
            scores = actor[state]
            action = 0 if scores[0] > scores[1] else 1
            next_state, reward, done = env_eval.step(action)
            total_reward += reward
            steps += 1
        total_rewards.append(total_reward)
        steps_list.append(steps)

    # Print visualizations
    print("Policy visualization:")
    for state in range(10):
        scores = actor[state]
        exp_scores = [math.exp(score) for score in scores]
        total = sum(exp_scores)
        probs = [exp_score / total for exp_score in exp_scores]
        action = 0 if probs[0] > probs[1] else 1
        print(f"State {state}: Action {action} (prob: {probs[action]:.2f})")

    print("\nValue function summary:")
    for state in range(10):
        print(f"State {state}: {critic[state]:.2f}")

    print("\nLearning progress summary:")
    avg_reward = sum(ep['total_reward'] for ep in episodes) / len(episodes)
    avg_steps = sum(ep['steps'] for ep in episodes) / len(episodes)
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.2f}")

    # Write CSV
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'steps', 'termination'])
        for ep in episodes:
            writer.writerow([
                ep['episode'],
                ep['total_reward'],
                ep['steps'],
                ep['termination']
            ])
