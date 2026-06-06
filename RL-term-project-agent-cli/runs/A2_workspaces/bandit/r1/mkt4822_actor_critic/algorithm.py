"""Actor-Critic reinforcement learning example using standard library only."""

import random
import time
import csv  # Added for CSV file operations
from typing import List, Dict

class Environment:
    """Deterministic environment for training Actor-Critic agent.
    
    The agent moves in a 1D space towards target state (5).
    Actions: 0 = left, 1 = right.
    Reward: +1 when reaching target, -0.1 otherwise.
    """
    def __init__(self, target=5):
        self.state = 0
        self.target = target

    def reset(self) -> int:
        """Reset environment to initial state."""
        self.state = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take an action and advance the environment.
        
        Args:
            action: Integer (0 or 1)
            
        Returns:
            next_state: Current state after action
            reward: Immediate reward
            done: Whether episode terminated
        """
        if action == 0:
            self.state -= 1
        else:
            self.state += 1
        
        reward = 1.0 if self.state == self.target else -0.1
        done = (self.state == self.target)
        
        return self.state, reward, done

class Actor:
    """Policy that selects actions based on state and threshold."""
    
    def __init__(self):
        self.threshold = 5.0  # Initial threshold for action selection
    
    def choose_action(self, state: int) -> int:
        """Select action based on current state and threshold.
        
        Args:
            state: Current environment state
            
        Returns:
            Action (0 or 1)
        """
        return 1 if state > self.threshold else 0

    def update(self, td_error: float, learning_rate: float):
        """Update the threshold using TD error.
        
        Args:
            td_error: Temporal difference error
            learning_rate: Step size for updates
        """
        self.threshold += learning_rate * td_error

class Critic:
    """Critic that estimates state values using TD(0)."""
    
    def __init__(self):
        self.values = {}  # State -> value estimate
    
    def get_value(self, state: int) -> float:
        """Get current value estimate for a state.
        
        Args:
            state: Current environment state
            
        Returns:
            Value estimate
        """
        return self.values.get(state, 0.0)
    
    def update(self, state: int, reward: float, next_state: int, done: bool, learning_rate: float):
        """Update value estimates using TD(0).
        
        Args:
            state: Current environment state
            reward: Immediate reward
            next_state: Next environment state
            done: Whether episode terminated
            learning_rate: Step size for updates
        """
        if state not in self.values:
            self.values[state] = 0.0
        
        if done:
            # Terminal state: update with immediate reward
            self.values[state] += learning_rate * (reward - self.values[state])
        else:
            # Non-terminal state: use next state's value
            self.values[state] += learning_rate * (reward + self.get_value(next_state) - self.values[state])

def train(env: Environment, actor: Actor, critic: Critic, episodes: int = 50, max_steps: int = 100):
    """Train the Actor-Critic agent for a number of episodes.
    
    Args:
        env: Environment instance
        actor: Actor policy
        critic: Critic value function
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
    """
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            action = actor.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # Update critic with current state, reward, next_state, done
            critic.update(state, reward, next_state, done, learning_rate=0.1)
            
            # Compute TD error for actor update
            td_error = reward + critic.get_value(next_state) - critic.get_value(state)
            actor.update(td_error, learning_rate=0.1)

            total_reward += reward
            state = next_state
            step_count += 1
        
        print(f"Episode {episode+1}: Reward={total_reward:.2f}, Steps={step_count}")

def evaluate(env: Environment, actor: Actor, episodes: int = 5):
    """Evaluate the trained agent on a test set.
    
    Args:
        env: Environment instance
        actor: Actor policy
        episodes: Number of evaluation episodes
        
    Returns:
        avg_reward: Average reward per episode
        steps_per_episode: Average steps per episode
    """
    total_reward = 0.0
    total_steps = 0
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        step_count = 0
        
        while not done:
            action = actor.choose_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            step_count += 1
            state = next_state
            
        total_steps += step_count
    
    avg_reward = total_reward / episodes
    steps_per_episode = total_steps / episodes
    return avg_reward, steps_per_episode

if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    
    env = Environment(target=5)
    actor = Actor()
    critic = Critic()
    
    print("Starting training...")
    train(env, actor, critic, episodes=50)
    
    print("\nEvaluating trained agent...")
    avg_reward, steps_per_episode = evaluate(env, actor)
    
    # Generate CSV report
    csv_path = "generation_report.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'steps'])
        
        # Simulate 50 episodes for the report
        for i in range(50):
            state = env.reset()
            done = False
            total_reward = 0.0
            step_count = 0
            
            while not done:
                action = actor.choose_action(state)
                next_state, reward, done = env.step(action)
                total_reward += reward
                step_count += 1
                state = next_state
                
            writer.writerow([i+1, total_reward, step_count])
    
    print(f"\nTraining complete. Generated report: {csv_path}")
