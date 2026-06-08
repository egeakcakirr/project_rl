"""Algorithm for a 1D grid environment using Actor-Critic RL.

This implementation focuses on boundary and edge-case handling for a 1D grid with states 0..9.
"""

from typing import List, Dict, Optional, Any, Tuple
import random
import csv
import sys
import math

class Environment:
    """Environment for a 1D grid with states 0..9.
    
    The agent starts at state 0, goal at state 9.
    Actions: 0 (left) or 1 (right).
    At state 0, action 0 keeps the agent at state 0.
    At state 9, action 1 keeps the agent at state 9 until the episode ends.
    Episode terminates when the agent reaches state 9 or after max_steps=35 steps.
    """
    def __init__(self, max_steps: int = 35):
        self.max_steps = max_steps
        self.current_state = 0
        self.steps = 0

    def reset(self) -> int:
        """Reset the environment to start state."""
        self.current_state = 0
        self.steps = 0
        return self.current_state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """Take a step in the environment.
        
        Args:
            action: 0 (left) or 1 (right)
        
        Returns:
            next_state: int, the next state
            reward: float, the reward for this step
            done: bool, whether the episode ended
        """
        if self.current_state == 9:
            return 9, 0, True  # Episode already ended

        if action == 0:
            next_state = max(0, self.current_state - 1)
        else:
            next_state = min(9, self.current_state + 1)

        reward = 10.0 if next_state == 9 else -1.0
        self.current_state = next_state
        self.steps += 1

        done = (next_state == 9) or (self.steps >= self.max_steps)
        return next_state, reward, done

def softmax(x: List[float]) -> List[float]:
    """Compute softmax probabilities from logits."""
    exp_x = [math.exp(i) for i in x]
    sum_exp = sum(exp_x)
    return [i / sum_exp for i in exp_x]

def main():
    random.seed(42)
    max_steps = 35
    num_train_episodes = 100
    num_eval_episodes = 10

    # Initialize critic and actor
    critic: Dict[int, float] = {s: 0.0 for s in range(10)}
    actor: Dict[int, List[float]] = {s: [0.0, 0.0] for s in range(10)}

    # Training loop
    training_results = []
    for episode in range(num_train_episodes):
        env = Environment(max_steps)
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            # Compute action probabilities using softmax
            probs = softmax(actor[state])
            action = random.choices([0, 1], weights=probs, k=1)[0]

            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

            # Compute TD error
            if done:
                next_value = 0.0
            else:
                next_value = critic[next_state]

            td_error = reward + 0.99 * next_value - critic[state]

            # Update critic
            critic[state] += 0.1 * td_error

            # Update actor
            for a in [0, 1]:
                prob = probs[a]
                actor[state][a] += 0.1 * td_error * prob

            state = next_state

        # Store training results
        termination = "goal" if total_reward == 10 else "max_steps"
        training_results.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'termination': termination
        })

    # Evaluation loop
    evaluation_results = []
    for _ in range(num_eval_episodes):
        env = Environment(max_steps)
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            # Greedy action selection
            probs = softmax(actor[state])
            action = 0 if probs[0] > probs[1] else 1

            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

            state = next_state

        termination = "goal" if total_reward == 10 else "max_steps"
        evaluation_results.append({
            'episode': len(evaluation_results) + 1,
            'total_reward': total_reward,
            'steps': steps,
            'termination': termination
        })

    # Generate CSV report
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'steps', 'termination'])
        for result in training_results + evaluation_results:
            writer.writerow([
                result['episode'],
                result['total_reward'],
                result['steps'],
                result['termination']
            ])

    # Print results
    print("Training episodes results:")
    for res in training_results:
        print(f"Episode {res['episode']}: Reward={res['total_reward']}, Steps={res['steps']}, Termination={res['termination']}")

    print("\nEvaluation results (10 greedy episodes):")
    for res in evaluation_results:
        print(f"Episode {res['episode']}: Reward={res['total_reward']}, Steps={res['steps']}, Termination={res['termination']}")

    # Print policy and value summaries
    print("\nPolicy summary (state → action probabilities):")
    for state in range(10):
        probs = softmax(actor[state])
        print(f"State {state}: Action 0: {probs[0]:.4f}, Action 1: {probs[1]:.4f}")

    print("\nValue summary (state → value):")
    for state in range(10):
        print(f"State {state}: {critic[state]:.4f}")

if __name__ == '__main__':
    main()
