"""Algorithm target file for mecha-agent-cli.

This file implements a minimal Actor-Critic reinforcement learning pipeline using standard library components.
"""

from __future__ import annotations
import random
import csv
import os
from typing import List, Dict, Tuple, Optional

class RandomWalkEnv:
    """Simple 1D random walk environment with discrete states and actions."""

    def __init__(self, max_steps: int = 10):
        self.max_steps: int = max_steps
        self.current_state: int = 0
        self.steps_taken: int = 0

    def reset(self) -> int:
        """Reset the environment to initial state."""
        self.current_state = 0
        self.steps_taken = 0
        return self.current_state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Execute one time step in the environment.
        
        Args:
            action: Integer action (0 for left, 1 for right)
            
        Returns:
            next_state: New state after action
            reward: Scalar reward value
            done: Boolean indicating episode termination
        """
        if self.steps_taken >= self.max_steps:
            return self.current_state, -1.0, True

        if action == 0:
            new_state = max(0, self.current_state - 1)
        else:
            new_state = min(2, self.current_state + 1)
        
        self.steps_taken += 1
        reward = -0.1
        done = False
        if new_state == 2:
            reward = 1.0
            done = True
        
        self.current_state = new_state
        return new_state, reward, done

class ActorCritic:
    """Actor-Critic reinforcement learning agent for tabular environments."""

    def __init__(self, env: RandomWalkEnv):
        self.env = env
        # Critic values (state -> value)
        self.critic_values: Dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}
        # Actor policy (state -> action)
        self.actor_policy: Dict[int, int] = {0: 0, 1: 0}

    def select_action(self, state: int) -> int:
        """Select action based on current policy."""
        return self.actor_policy.get(state, 0)

    def update_critic(self, state: int, next_state: int, reward: float, done: bool):
        """Update critic using TD(0) method with learning rate 0.1."""
        if done:
            delta = reward - self.critic_values[state]
        else:
            delta = (reward + 0.9 * self.critic_values[next_state]) - self.critic_values[state]
        
        self.critic_values[state] += 0.1 * delta

def train_actor_critic(env: RandomWalkEnv, num_episodes: int = 100) -> List[float]:
    """Train the Actor-Critic agent with specified episodes."""
    ac = ActorCritic(env)
    rewards_history: List[float] = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action = ac.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Update critic
            ac.update_critic(state, next_state, reward, done)
            
            # Update actor policy for current state using critic values of possible transitions
            if not done:
                action0_next = max(0, state - 1)
                action1_next = min(2, state + 1)
                
                best_action = 0 if ac.critic_values[action0_next] > ac.critic_values[action1_next] else 1
                ac.actor_policy[state] = best_action
            
            total_reward += reward
            state = next_state
        
        rewards_history.append(total_reward)
    
    return rewards_history

def evaluate_policy(env: RandomWalkEnv, ac: ActorCritic, num_eval_episodes: int = 5) -> float:
    """Evaluate policy performance across multiple episodes."""
    total_rewards = 0.0
    
    for _ in range(num_eval_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = ac.select_action(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
        
        total_rewards += episode_reward
    
    return total_rewards / num_eval_episodes

def generate_report_csv(rewards_history: List[float], eval_avg_reward: float):
    """Generate CSV report with training and evaluation metrics."""
    filename = "generation_report.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Total Reward'])
        
        for episode, reward in enumerate(rewards_history, start=1):
            writer.writerow([episode, reward])
        
        writer.writerow(['Evaluation Average Reward', eval_avg_reward])

if __name__ == "__main__":
    # Seed for deterministic behavior
    random.seed(42)
    
    env = RandomWalkEnv()
    num_episodes = 100
    
    # Train agent
    rewards_history = train_actor_critic(env, num_episodes)
    
    # Evaluate policy
    eval_avg_reward = evaluate_policy(env, ActorCritic(env), 5)
    
    # Generate report
    generate_report_csv(rewards_history, eval_avg_reward)
    
    # Print summary metrics
    avg_train_reward = sum(rewards_history) / len(rewards_history)
    print(f"Training completed: {num_episodes} episodes")
    print(f"Average training reward: {avg_train_reward:.2f}")
    print(f"Evaluation average reward: {eval_avg_reward:.2f}")
