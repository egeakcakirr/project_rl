"""Actor-Critic reinforcement learning implementation using standard library only."""

import csv
import sys
from random import seed as random_seed
from typing import List, Dict, Optional, Tuple

class SimpleEnvironment:
    """Simple 1D environment for demonstration with deterministic dynamics."""
    
    def __init__(self):
        self.position = 0.0
        self.target = 1.0
        self.max_steps = 50
    
    def reset(self) -> float:
        """Reset environment to initial state and return starting position."""
        self.position = 0.0
        return self.position
    
    def step(self, action: int) -> Tuple[float, float, bool]:
        """
        Execute one time step in the environment.
        
        Args:
            action: 0 for left movement, 1 for right movement
            
        Returns:
            new_position: Current position after action
            reward: Immediate reward (1.0 if target reached, else 0.0)
            done: Whether episode terminated
        """
        if action == 0:
            self.position -= 0.1
        else:
            self.position += 0.1
        
        # Clamp position to [0.0, 1.0] range
        self.position = max(0.0, min(self.position, 1.0))
        
        reward = 1.0 if self.position >= self.target else 0.0
        done = (self.position >= self.target) or (self.position <= 0.0)
        return self.position, reward, done

class ActorCritic:
    """Tabular implementation of Actor-Critic algorithm for demonstration."""
    
    def __init__(self, env):
        self.env = env
        self.critic_values: Dict[float, float] = {}
        self.actor_policy: Dict[float, int] = {}
    
    def reset(self):
        """Reset policy and value estimates to initial state."""
        self.critic_values.clear()
        self.actor_policy.clear()
    
    def update_critic(self, state: float, next_state: float, reward: float, done: bool) -> None:
        """
        Update critic values using TD(1) error.
        
        Args:
            state: Current state
            next_state: Next state after action
            reward: Immediate reward
            done: Episode termination flag
        """
        if done:
            td_error = reward - self.critic_values.get(next_state, 0.0)
        else:
            next_value = self.critic_values.get(next_state, 0.0)
            td_error = reward + next_value - self.critic_values.get(state, 0.0)
        
        # Apply learning rate (0.1) to update critic values
        self.critic_values[state] = self.critic_values.get(state, 0.0) + 0.1 * td_error
    
    def update_actor(self, state: float, action: int, reward: float, next_state: float, done: bool) -> None:
        """
        Update actor policy using REINFORCE method.
        
        Args:
            state: Current state
            action: Chosen action
            reward: Immediate reward
            next_state: Next state after action
            done: Episode termination flag
        """
        if state in self.critic_values:
            # Simplified policy gradient update (demonstration only)
            grad = 0.1 * (reward - self.critic_values[state])
            self.actor_policy[state] = int(self.actor_policy.get(state, 0) + grad)
    
    def act(self, state: float) -> int:
        """Select action based on current policy."""
        return self.actor_policy.get(state, 0)

def train_agent(env: SimpleEnvironment, num_episodes: int = 100) -> List[float]:
    """
    Train agent using Actor-Critic algorithm.
    
    Args:
        env: Environment instance
        num_episodes: Number of training episodes
        
    Returns:
        List of total rewards per episode
    """
    agent = ActorCritic(env)
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Update critic and actor
            agent.update_critic(state, next_state, reward, done)
            agent.update_actor(state, action, reward, next_state, done)
            
            state = next_state
        
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes} completed with reward: {total_reward:.2f}")
    
    return total_rewards

def evaluate_agent(env: SimpleEnvironment, agent: ActorCritic, num_eval_episodes: int = 5) -> float:
    """
    Evaluate trained agent on new episodes.
    
    Args:
        env: Environment instance
        agent: Trained actor-critic model
        num_eval_episodes: Number of evaluation episodes
        
    Returns:
        Average reward across evaluation episodes
    """
    total_rewards = []
    
    for _ in range(num_eval_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            
        total_rewards.append(total_reward)
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average reward after evaluation: {avg_reward:.2f}")
    return avg_reward

if __name__ == "__main__":
    # Seed for deterministic behavior
    random_seed(42)
    
    env = SimpleEnvironment()
    agent = ActorCritic(env)
    
    print("Starting training...")
    total_rewards = train_agent(env, num_episodes=100)
    
    print("\nEvaluating agent...")
    evaluate_agent(env, agent)
    
    # Generate CSV report with training metrics
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward'])
        
        for i, reward in enumerate(total_rewards):
            writer.writerow([i + 1, round(reward, 2)])
    
    print("\nTraining completed. Report generated: generation_report.csv")
