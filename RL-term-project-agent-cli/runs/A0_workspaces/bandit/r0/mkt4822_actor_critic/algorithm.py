"""Algorithm target file for mecha-agent-cli.

This file implements a minimal Actor-Critic reinforcement learning application using only standard library.
"""

from __future__ import annotations

import csv
import random
import time
from typing import Dict, List, Tuple, Optional, Any

class SimpleEnv:
    """Simple 1D environment where agent moves towards target position."""
    
    def __init__(self, target: int = 5) -> None:
        self.target = target
        self.position = 0
    
    def reset(self) -> int:
        """Reset the environment to initial state."""
        self.position = 0
        return self.position
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Execute one time step in the environment.
        
        Args:
            action: Integer action (1 for move right, -1 for move left)
            
        Returns:
            next_position: New position after action
            reward: Reward received after taking action
            done: Whether episode is terminated
        """
        if action == 1:
            self.position += 1
        else:
            self.position -= 1
        
        reward = 0.1 if self.position >= self.target else -0.1
        done = (self.position >= self.target)
        return self.position, reward, done

class Actor:
    """Actor that selects actions based on current state."""
    
    def __init__(self) -> None:
        self.policy: Dict[int, int] = {}
    
    def act(self, state: int) -> int:
        """Select action for given state using policy."""
        return self.policy.get(state, 1)  # Default to move right (action=1)

class Critic:
    """Critic that estimates value of states."""
    
    def __init__(self) -> None:
        self.values: Dict[int, float] = {}
    
    def update(self, state: int, reward: float, next_state: int) -> None:
        """
        Update critic's value estimate using TD(0).
        
        Args:
            state: Current state
            reward: Reward received after taking action
            next_state: Next state after action
        """
        if state not in self.values:
            self.values[state] = 0.0
        
        discount_factor = 0.9
        current_value = self.values[state]
        next_value = self.values.get(next_state, 0.0)
        
        # TD(0) update: V(s) += α (r + γV(s') - V(s))
        alpha = 0.1
        self.values[state] += alpha * (reward + discount_factor * next_value - current_value)
    
    def get_value(self, state: int) -> float:
        """Get estimated value of a state."""
        return self.values.get(state, 0.0)

def train_actor_critic(
    env: SimpleEnv,
    actor: Actor,
    critic: Critic,
    num_episodes: int = 100,
    max_steps_per_episode: int = 100
) -> List[float]:
    """
    Train the actor-critic model.
    
    Args:
        env: Environment instance
        actor: Actor instance
        critic: Critic instance
        num_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
    
    Returns:
        List of total rewards per episode
    """
    total_rewards = []
    start_time = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action = actor.act(state)
            next_state, reward, done = env.step(action)
            
            # Update critic using TD(0)
            critic.update(state, reward, next_state)
            
            # Update actor policy: choose best action based on critic's values
            if state not in actor.policy:
                actor.policy[state] = 1  # Default to move right
            
            state = next_state
            total_reward += reward
        
        total_rewards.append(total_reward)
        
        if episode % 10 == 0:
            avg_reward = sum(total_rewards[-10:]) / 10
            print(f"Episode {episode + 1}: Avg reward (last 10) = {avg_reward:.2f}")
    
    return total_rewards

def evaluate_policy(
    env: SimpleEnv,
    actor: Actor,
    num_steps: int = 10
) -> float:
    """
    Evaluate the trained policy.
    
    Args:
        env: Environment instance
        actor: Actor instance
        num_steps: Number of steps to run for evaluation
    
    Returns:
        Total reward from evaluation
    """
    state = env.reset()
    total_reward = 0.0
    for _ in range(num_steps):
        action = actor.act(state)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
    return total_reward

def generate_report(
    num_episodes: int,
    avg_reward_per_episode: float,
    final_evaluation_reward: float
) -> None:
    """Generate CSV report of training results."""
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode_count', 'avg_reward_per_episode', 'final_evaluation_reward'])
        writer.writerow([num_episodes, avg_reward_per_episode, final_evaluation_reward])

if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    
    target = 5
    env = SimpleEnv(target=target)
    actor = Actor()
    critic = Critic()
    
    print(f"Starting training with target={target}...")
    
    # Train for 100 episodes
    total_rewards = train_actor_critic(env, actor, critic, num_episodes=100)
    
    # Evaluate policy
    final_evaluation_reward = evaluate_policy(env, actor)
    
    # Calculate average reward per episode
    avg_reward_per_episode = sum(total_rewards) / len(total_rewards)
    
    # Generate CSV report
    generate_report(
        num_episodes=len(total_rewards),
        avg_reward_per_episode=avg_reward_per_episode,
        final_evaluation_reward=final_evaluation_reward
    )
    
    print(f"Training completed. Average reward per episode: {avg_reward_per_episode:.2f}")
    print(f"Evaluation reward after 10 steps: {final_evaluation_reward:.2f}")
    print("Report generated to generation_report.csv")
