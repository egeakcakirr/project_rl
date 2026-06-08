import random
import time
import math
import os
import csv
from typing import List, Tuple

class SimpleEnv:
    """A simple 1D environment for demonstration.

    The agent starts at state 0 and aims to reach state 3.
    Actions: 0 (move left), 1 (move right).
    Reward: +1 when reaching goal, -0.1 otherwise.
    """
    def __init__(self):
        self.goal = 3
        self.state = 0

    def reset(self) -> int:
        """Reset environment to initial state."""
        self.state = 0
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """Take an action and advance the environment.

        Args:
            action: Integer action (0 for left, 1 for right)

        Returns:
            tuple: (next_state, reward, done)
        """
        if action == 0 and self.state > 0:
            self.state -= 1
        elif action == 1 and self.state < self.goal:
            self.state += 1

        reward = 1.0 if self.state == self.goal else -0.1
        done = self.state >= self.goal
        return self.state, reward, done


class ActorCritic:
    """Actor-Critic model for reinforcement learning.

    This simple implementation uses linear weights for both actor and critic.
    """
    def __init__(self, env: SimpleEnv):
        self.env = env
        # Initialize weights for actor (w1, w2) and critic (v)
        self.actor_weights: List[float] = [0.0, 0.0]
        self.critic_weights: float = 0.0

    def get_action(self, state: int) -> int:
        """Select action using the current policy.

        Args:
            state: Current environment state

        Returns:
            Integer action (0 or 1)
        """
        # Calculate probability of taking action 0
        prob_0 = 1 / (1 + math.exp(-(self.actor_weights[0] * state + self.actor_weights[1])))
        return 0 if random.random() < prob_0 else 1

    def update_actor(self, state: int, action: int, reward: float):
        """Update actor weights using policy gradient.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
        """
        lr = 0.1
        # Simplified TD error (for demonstration)
        td_error = reward - self.critic_weights * state
        self.actor_weights[0] += lr * td_error * state
        self.actor_weights[1] += lr * td_error

    def update_critic(self, state: int, next_state: int):
        """Update critic weights using TD error.

        Args:
            state: Current state
            next_state: State after action
        """
        pass  # For simplicity, we don't use this in the demo


def train_actor_critic(env: SimpleEnv, episodes: int = 10) -> List[float]:
    """Train the actor-critic model for a given number of episodes.

    Args:
        env: Environment instance
        episodes: Number of training episodes

    Returns:
        List of total rewards per episode.
    """
    actor_critic = ActorCritic(env)
    rewards_per_episode: List[float] = []
    random.seed(42)  # Ensure reproducibility
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = actor_critic.get_action(state)
            next_state, reward, done = env.step(action)
            actor_critic.update_actor(state, action, reward)
            total_reward += reward
            state = next_state
        rewards_per_episode.append(total_reward)
    return rewards_per_episode


if __name__ == "__main__":
    # Initialize environment with deterministic seed
    random.seed(42)
    env = SimpleEnv()
    
    # Train for 10 episodes (small but non-trivial)
    actor_critic = ActorCritic(env)
    rewards_per_episode = train_actor_critic(env, 10)
    
    # Evaluate on a test set of 5 episodes
    eval_rewards = []
    for _ in range(5):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = actor_critic.get_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
        eval_rewards.append(total_reward)
    
    avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
    
    # Generate CSV report
    csv_path = "generation_report.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward'])
        for i, reward in enumerate(rewards_per_episode):
            writer.writerow([i + 1, round(reward, 2)])
        writer.writerow(['evaluation', 'average_reward'])
        writer.writerow([None, round(avg_eval_reward, 2)])
    
    print(f"Training completed. Evaluation average reward: {avg_eval_reward:.2f}. Report saved to {csv_path}")
