"""Actor-Critic reinforcement learning example using standard library only."""

from __future__ import annotations

import random
import os
from collections import defaultdict
from typing import List, Dict, Any

class SimpleEnvironment:
    """A simple 1D grid environment for demonstration.

    The agent moves between states 0-2 with actions: left (0) or right (1).
    Reward is +1 when reaching state 2 (goal), else -0.1 per step.
    Episode terminates after max_steps steps if goal not reached.
    """

    def __init__(self):
        self.state = 0
        self.max_steps = 5
        self.steps = 0

    def reset(self) -> int:
        """Reset the environment to initial state."""
        self.state = 0
        self.steps = 0
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """Take an action and return next state, reward, done status."""
        if action == 0:
            new_state = max(0, self.state - 1)
        else:
            new_state = min(2, self.state + 1)
        
        # Update step count
        self.steps += 1
        
        reward = 1.0 if new_state == 2 else -0.1
        done = (new_state == 2) or (self.steps >= self.max_steps)
        
        return new_state, reward, done

class ActorCritic:
    """Actor-Critic model for reinforcement learning.

    This implementation uses a tabular critic and a stochastic policy.
    """

    def __init__(self, env: SimpleEnvironment):
        self.env = env
        self.critic_values: Dict[int, float] = defaultdict(float)
        # Stochastic policy (50% chance for each action)
        self.policy = lambda state: 0 if random.random() < 0.5 else 1

    def train(self, episodes: int = 100, gamma: float = 0.9, alpha: float = 0.1) -> List[float]:
        """Train the actor-critic model for a given number of episodes.

        Args:
            episodes: Number of training episodes.
            gamma: Discount factor for future rewards.
            alpha: Learning rate for critic updates.

        Returns:
            List of total rewards per episode.
        """
        total_rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done = self.env.step(action)
                # Update critic using TD(0) update rule
                old_value = self.critic_values[state]
                new_value = reward + gamma * (self.critic_values[next_state] if next_state in self.critic_values else 0.0)
                self.critic_values[state] = old_value + alpha * (new_value - old_value)
                total_reward += reward
                state = next_state
            total_rewards.append(total_reward)
        return total_rewards

    def evaluate(self) -> float:
        """Evaluate the trained agent in a single episode.

        Returns:
            Total reward achieved.
        """
        state = self.env.reset()
        total_reward = 0
        done = False
        while not done:
            action = self.policy(state)
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            state = next_state
        return total_reward

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    
    env = SimpleEnvironment()
    ac = ActorCritic(env)
    
    # Train for 10 episodes (small but non-trivial)
    total_rewards = ac.train(episodes=10, gamma=0.9, alpha=0.1)
    
    # Evaluate the agent
    final_reward = ac.evaluate()
    
    # Generate CSV report with training metrics
    with open('generation_report.csv', 'w') as f:
        f.write("episode,average_reward\n")
        for i, reward in enumerate(total_rewards):
            avg_reward = sum(total_rewards[:i+1]) / (i+1)
            f.write(f"{i+1},{avg_reward:.2f}\n")
    
    print(f"Training complete. Final reward: {final_reward}")
