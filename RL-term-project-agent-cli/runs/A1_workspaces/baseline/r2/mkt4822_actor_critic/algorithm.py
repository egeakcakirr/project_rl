"""Algorithm target file for mecha-agent-cli.

This file implements a simple Actor-Critic reinforcement learning application using only standard library modules.
"""

from __future__ import annotations
import random
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class SimpleEnvironment:
    """A deterministic 1D environment where the agent moves left or right to reach the goal state."""
    
    def __init__(self, goal_state: int = 2):
        self.goal_state: int = goal_state
        self.current_state: int = 0
    
    def reset(self) -> int:
        """Reset the environment to the initial state."""
        self.current_state = 0
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take a single step in the environment.
        
        Args:
            action: Integer action (0 for left, 1 for right)
            
        Returns:
            next_state: New state after taking action
            reward: Reward received for this step
            done: Whether the episode terminated
        """
        if action == 0:
            next_state = max(0, self.current_state - 1)
        else:
            next_state = min(self.goal_state, self.current_state + 1)
        
        reward = 1.0 if next_state == self.goal_state else -0.1
        done = (next_state == self.goal_state)
        return next_state, reward, done

class Actor:
    """Deterministic policy that maps states to actions."""
    
    def __init__(self):
        self.policy: Dict[int, int] = {}
    
    def select_action(self, state: int) -> int:
        """Select an action based on the current policy."""
        return self.policy.get(state, 0)
    
    def update_policy(self, state: int, action: int, value: float):
        """Update the policy for a given state with new action and value."""
        self.policy[state] = action

class Critic:
    """Value function estimator that maps states to estimated values."""
    
    def __init__(self):
        self.values: Dict[int, float] = defaultdict(float)
    
    def get_value(self, state: int) -> float:
        """Get the current value estimate for a state."""
        return self.values[state]
    
    def update_value(self, state: int, next_state: int, reward: float, gamma: float = 0.9):
        """Update the value of the current state using TD error."""
        alpha = 0.1
        next_value = self.get_value(next_state)
        td_error = reward + gamma * next_value - self.get_value(state)
        self.values[state] += alpha * td_error

def train_actor_critic(
    env: SimpleEnvironment,
    actor: Actor,
    critic: Critic,
    num_episodes: int = 10
) -> List[float]:
    """Train the actor and critic for a specified number of episodes.
    
    Args:
        env: Environment instance
        actor: Actor instance
        critic: Critic instance
        num_episodes: Number of training episodes
    
    Returns:
        List of total rewards per episode
    """
    gamma = 0.9
    rewards_per_episode = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            action = actor.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Update critic using TD error
            critic.update_value(state, next_state, reward, gamma)
            
            state = next_state
            total_reward += reward
            
        rewards_per_episode.append(total_reward)
        
        # Update actor policy after each episode (simplified rule)
        for state in range(env.goal_state + 1):
            action_values = []
            for action in [0, 1]:
                next_state, _, _ = env.step(action)
                value = critic.get_value(next_state)
                action_values.append(value)
            
            best_action = 0 if action_values[0] > action_values[1] else 1
            actor.update_policy(state, best_action, critic.get_value(state))
    
    return rewards_per_episode

def evaluate(
    env: SimpleEnvironment,
    actor: Actor,
    critic: Critic
) -> float:
    """Evaluate the trained agent on a single test episode.
    
    Args:
        env: Environment instance
        actor: Actor instance
        critic: Critic instance
    
    Returns:
        Total reward from one evaluation episode
    """
    state = env.reset()
    total_reward = 0.0
    done = False
    
    while not done:
        action = actor.select_action(state)
        next_state, reward, done = env.step(action)
        
        state = next_state
        total_reward += reward
        
    return total_reward

if __name__ == "__main__":
    # Set a fixed seed for reproducibility
    random.seed(42)
    
    # Initialize environment and agents
    env = SimpleEnvironment(goal_state=2)
    actor = Actor()
    critic = Critic()
    
    # Train the agent
    print("Starting training...")
    rewards_per_episode = train_actor_critic(env, actor, critic, num_episodes=5)
    
    # Evaluate after training
    test_reward = evaluate(env, actor, critic)
    
    # Generate CSV report
    with open('generation_report.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Total Reward'])
        for episode_idx, reward in enumerate(rewards_per_episode):
            writer.writerow([episode_idx + 1, reward])
    
    print("\nTraining complete!")
    print(f"Rewards per episode: {rewards_per_episode}")
    print(f"Evaluation reward: {test_reward}")
    print("Report saved to generation_report.csv")
