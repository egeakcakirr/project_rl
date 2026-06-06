from __future__ import annotations
import random
import time
import csv
import sys
import math

random.seed(42)

class GridWorld:
    def __init__(self):
        self.state: int = 0
        self.steps: int = 0
        self.max_steps: int = 30

    def reset(self) -> int:
        self.state = 0
        self.steps = 0
        return self.state

    def step(self, action: int) -> tuple[int, int, bool]:
        self.steps += 1
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(9, self.state + 1)
        
        reward = 10 if self.state == 9 else -1
        done = self.state == 9 or self.steps >= self.max_steps
        return self.state, reward, done

gamma = 0.9
alpha = 0.1

V = {s: 0.0 for s in range(10)}
theta = {s: [0.0, 0.0] for s in range(10)}

if __name__ == '__main__':
    # Train for 80 episodes
    results = []
    for episode in range(80):
        env = GridWorld()
        s = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Compute softmax probabilities
            probs = [math.exp(theta[s][a]) for a in range(2)]
            probs = [p / sum(probs) for p in probs]
            
            # Select action
            a = random.choices([0, 1], weights=probs)[0]
            
            s_next, r, done = env.step(a)
            total_reward += r
            steps += 1
            
            # Compute TD error
            delta = r + gamma * V[s_next] - V[s]
            
            # Update critic
            V[s] += alpha * delta
            
            # Update actor
            for a in [0, 1]:
                theta[s][a] += alpha * delta * (probs[a] - 1)
            
            s = s_next
        
        # Determine termination reason
        termination = "goal" if s == 9 else "timeout"
        results.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "termination": termination
        })
        print(f"Episode {episode}: Reward={total_reward}, Steps={steps}, Termination={termination}")

    # Evaluate for 10 episodes
    eval_results = []
    for i in range(10):
        env = GridWorld()
        s = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Greedy action selection
            next_states = []
            for a in [0, 1]:
                if a == 0:
                    ns = max(0, s - 1)
                else:
                    ns = min(9, s + 1)
                next_states.append(ns)
            
            # Choose action with highest value
            action_index = 0
            max_value = -float('inf')
            for a in [0, 1]:
                if V[next_states[a]] > max_value:
                    max_value = V[next_states[a]]
                    action_index = a
            
            a = action_index
            s_next, r, done = env.step(a)
            total_reward += r
            steps += 1
        
        termination = "goal" if s == 9 else "timeout"
        eval_results.append({
            "episode": 80 + i,
            "total_reward": total_reward,
            "steps": steps,
            "termination": termination
        })
    
    # Generate CSV
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'steps', 'termination'])
        for res in results:
            writer.writerow([res['episode'], res['total_reward'], res['steps'], res['termination']])
    
    # Textual policy visualization
    print("\nPolicy visualization:")
    for s in range(10):
        probs = [math.exp(theta[s][a]) for a in range(2)]
        probs = [p / sum(probs) for p in probs]
        print(f"State {s}: left ({probs[0]:.2f}), right ({probs[1]:.2f})")

    # Print evaluation results
    print("\nEvaluation results (10 episodes):")
    for i, res in enumerate(eval_results):
        print(f"Episode {80 + i}: Reward={res['total_reward']}, Steps={res['steps']}, Termination={res['termination']}")
