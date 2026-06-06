import csv
import random
from collections import defaultdict
import sys

def main():
    random.seed(42)
    gamma = 0.95
    alpha = 0.1

    class GridWorldEnv:
        def __init__(self, max_steps: int = 35):
            self.state: int = 0
            self.steps: int = 0
            self.max_steps: int = max_steps

        def step(self, action: int) -> tuple[int, int, bool]:
            """Transition helper for the grid world environment."""
            valid_action = 0 if action not in [0, 1] else action
            if valid_action == 0:
                next_state = max(0, self.state - 1)
            else:
                next_state = min(9, self.state + 1)
            self.state = next_state
            self.steps += 1
            done = self.state == 9 or self.steps >= self.max_steps
            reward = 10 if done and self.state == 9 else -1
            return next_state, reward, done

    actor: dict[int, list[float]] = defaultdict(lambda: [0.5, 0.5])
    critic: dict[int, float] = defaultdict(float)

    results = []
    episodes = 100

    for episode in range(episodes):
        env = GridWorldEnv()
        state = env.state
        total_reward = 0
        steps = 0
        done = False

        while not done:
            p0, p1 = actor[state]
            action = random.choices([0, 1], weights=[p0, p1], k=1)[0]

            if random.random() < 0.1:
                action = random.randint(0, 3)

            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

            next_value = critic[next_state] if not done else 0.0
            td_error = reward + gamma * next_value - critic[state]
            critic[state] += alpha * td_error

            if action == 0:
                actor[state][0] += alpha * td_error
            else:
                actor[state][1] += alpha * td_error

            total_prob = actor[state][0] + actor[state][1]
            actor[state][0] = actor[state][0] / total_prob
            actor[state][1] = actor[state][1] / total_prob

        results.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'termination': 'goal' if state == 9 else 'max_steps'
        })

    eval_results = []
    for _ in range(10):
        env = GridWorldEnv()
        state = env.state
        total_reward = 0
        steps = 0
        done = False
        while not done:
            p0, p1 = actor[state]
            action = 0 if p0 > p1 else 1
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
        eval_results.append({
            'episode': f'eval_{_}',
            'total_reward': total_reward,
            'steps': steps,
            'termination': 'goal' if next_state == 9 else 'max_steps'
        })

    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'total_reward', 'steps', 'termination'])
        writer.writeheader()
        for res in results:
            writer.writerow(res)

    print("Training results:")
    for res in results:
        print(f"Episode {res['episode']}: Reward={res['total_reward']}, Steps={res['steps']}, Termination={res['termination']}")

    print("\nPolicy summary:")
    for state in range(10):
        p0, p1 = actor[state]
        print(f"State {state}: Action 0 ({p0:.2f}), Action 1 ({p1:.2f})")

    print("\nValue summary:")
    for state in range(10):
        print(f"State {state}: Value={critic[state]:.2f}")

if __name__ == '__main__':
    main()
