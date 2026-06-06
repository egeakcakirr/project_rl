"""Algorithm for a simple 1D grid world Actor-Critic RL agent.

This script implements a tabular Actor-Critic reinforcement learning agent for a 1D grid world with states 0-9.
"""

from __future__ import annotations

import random
import csv
import sys
import time

class GridWorld:
    """Simple 1D grid world environment with states 0-9."""
    def __init__(self, max_steps: int = 35):
        self.max_steps = max_steps
        self.goal = 9

    def step(self, state: int, action: str) -> tuple[int, int, bool]:
        """Execute an action in the environment and return next state, reward, done."""
        if action == 'left':
            next_state = max(0, state - 1)
        elif action == 'right':
            next_state = min(9, state + 1)
        else:
            next_state = state
        reward = 10 if next_state == self.goal else -1
        done = next_state == self.goal
        return next_state, reward, done

class ActorCritic:
    """Tabular Actor-Critic model for the grid world."""
    def __init__(self, num_states: int = 10):
        self.num_states = num_states
        self.policy = ['right'] * num_states  # Initial policy: all right
        self.critic = [0.0] * num_states  # Initial critic values

    def update_critic(self, state: int, next_state: int, reward: int, gamma: float = 0.9, alpha: float = 0.1):
        """Update critic value using TD error."""
        td_error = reward + gamma * self.critic[next_state] - self.critic[state]
        self.critic[state] += alpha * td_error

    def update_policy(self, state: int, next_state: int, reward: int, gamma: float = 0.9, alpha: float = 0.1):
        """Update policy to select action with highest critic value for next state."""
        left_value = self.critic[max(0, state - 1)]
        right_value = self.critic[min(9, state + 1)]
        self.policy[state] = 'left' if left_value > right_value else 'right'

def train_actor_critic(env: GridWorld, agent: ActorCritic, num_episodes: int = 100, max_steps: int = 35):
    """Train the Actor-Critic agent for a given number of episodes."""
    gamma = 0.9
    alpha = 0.1
    total_rewards = []
    for episode in range(num_episodes):
        state = 0
        total_reward = 0
        steps = 0
        while steps < max_steps:
            action = agent.policy[state]
            next_state, reward, done = env.step(state, action)
            total_reward += reward
            steps += 1
            if done:
                break
            agent.update_critic(state, next_state, reward, gamma, alpha)
        total_rewards.append(total_reward)
        agent.update_policy(state, next_state, reward, gamma, alpha)
    return total_rewards

def evaluate_agent(env: GridWorld, agent: ActorCritic, num_episodes: int = 10):
    """Evaluate the trained agent on a new set of episodes."""
    total_rewards = []
    for _ in range(num_episodes):
        state = 0
        total_reward = 0
        steps = 0
        while steps < 35:
            action = agent.policy[state]
            next_state, reward, done = env.step(state, action)
            total_reward += reward
            steps += 1
            if done:
                break
        total_rewards.append(total_reward)
    return total_rewards

def visualize_policy(agent: ActorCritic):
    """Generate textual visualization of the policy."""
    print("\nPolicy (state -> action):")
    for state in range(10):
        print(f"State {state}: {agent.policy[state]}")

def visualize_critic(agent: ActorCritic):
    """Generate textual visualization of the critic values."""
    print("\nCritic values (state -> value):")
    for state in range(10):
        print(f"State {state}: {agent.critic[state]:.2f}")

def generate_report_csv(total_rewards: list[float], evaluation_rewards: list[float]):
    """Generate CSV report of training performance."""
    min_len = min(len(total_rewards), len(evaluation_rewards))
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Total Reward', 'Evaluation Reward'])
        for i in range(min_len):
            writer.writerow([i + 1, total_rewards[i], evaluation_rewards[i]])

if __name__ == '__main__':
    # Set deterministic seed for reproducibility
    random.seed(42)
    
    # Initialize environment and agent
    env = GridWorld()
    agent = ActorCritic()
    
    # Train the agent
    print("Starting training...")
    total_rewards = train_actor_critic(env, agent, num_episodes=100)
    
    # Evaluate the agent
    print("\nEvaluating agent...")
    evaluation_rewards = evaluate_agent(env, agent, num_episodes=10)
    
    # Visualize policy and critic
    visualize_policy(agent)
    visualize_critic(agent)
    
    # Generate CSV report
    generate_report_csv(total_rewards, evaluation_rewards)
    
    print("\nTraining completed. Report generated as generation_report.csv")
