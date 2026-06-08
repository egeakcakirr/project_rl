"""Algorithm target file for mecha-agent-cli.

This file implements a minimal Actor-Critic reinforcement learning agent using standard library only.
"""

from typing import List, Dict, Tuple, Optional
import random
import csv
import os

class GridWorldEnv:
    """Simple 1D grid world environment with deterministic state transitions."""
    
    def __init__(self, goal: int = 2):
        self.goal = goal
        self.state = 0
    
    def reset(self) -> int:
        """Reset environment to initial state (state=0)."""
        self.state = 0
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Execute one time step in the environment.
        
        Args:
            action: Integer action (-1 for left, +1 for right)
            
        Returns:
            next_state: New state after action
            reward: Immediate reward
            done: Whether episode terminated
        """
        next_state = self.state + action
        reward = -1.0
        done = False
        
        # Clamp state to valid range [0, goal]
        if next_state < 0:
            next_state = 0
        elif next_state > self.goal:
            next_state = self.goal
            
        # Check for terminal state (goal reached)
        if next_state == self.goal:
            reward = 1.0
            done = True
        
        self.state = next_state
        return next_state, reward, done

class ActorCritic:
    """Actor-Critic reinforcement learning agent using tabular methods."""
    
    def __init__(self, env: GridWorldEnv):
        """
        Initialize actor-critic model.
        
        Args:
            env: Grid world environment instance
        """
        self.env = env
        self.V = [0.0] * (env.goal + 1)  # Value function for each state
        self.epsilon = 0.2  # Exploration rate
        self.gamma = 0.9   # Discount factor
    
    def select_action(self, state: int) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Action (-1 for left, +1 for right)
        """
        if random.random() < self.epsilon:
            return random.choice([-1, 1])  # Random exploration
        else:
            # Move toward goal when possible
            return 1 if state < self.env.goal else -1
    
    def update_critic(self, state: int, next_state: int, reward: float):
        """Update critic value using TD error."""
        td_error = reward + self.gamma * self.V[next_state] - self.V[state]
        self.V[state] += 0.1 * td_error  # Small learning rate
    
    def train_episode(self) -> float:
        """
        Train for one episode.
        
        Returns:
            Total reward accumulated in the episode
        """
        total_reward = 0.0
        state = self.env.reset()
        done = False
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.env.step(action)
            self.update_critic(state, next_state, reward)
            total_reward += reward
            state = next_state
            
        return total_reward
    
    def train(self, num_episodes: int) -> List[float]:
        """
        Train agent for specified number of episodes.
        
        Args:
            num_episodes: Number of training episodes
            
        Returns:
            List of rewards per episode
        """
        rewards = []
        for _ in range(num_episodes):
            reward = self.train_episode()
            rewards.append(reward)
        return rewards

if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    
    env = GridWorldEnv(goal=2)
    ac = ActorCritic(env)
    
    # Train for 100 episodes
    training_rewards = ac.train(num_episodes=100)
    
    # Evaluate on test episodes (5 episodes)
    test_rewards = []
    for _ in range(5):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = ac.select_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
        test_rewards.append(total_reward)
    
    # Generate CSV report
    csv_path = os.path.join(os.getcwd(), "generation_report.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Total Reward'])
        for i, reward in enumerate(training_rewards):
            writer.writerow([i+1, round(reward, 2)])
    
    # Print training progress
    print("\nTraining complete. Generated report: generation_report.csv")
    print(f"Test average reward: {sum(test_rewards) / len(test_rewards):.2f}")
    print("\nTraining progress:")
    for i, reward in enumerate(training_rewards):
        print(f"Episode {i+1}: Reward = {reward:.2f}")
