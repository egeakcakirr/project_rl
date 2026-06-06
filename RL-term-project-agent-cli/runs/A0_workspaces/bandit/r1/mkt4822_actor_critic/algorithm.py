from __future__ import annotations
import csv
import os
import random

class SimpleEnv:
    """A simple deterministic environment with two states and two actions.
    
    The agent starts at state 0. Action 0 moves to state 1 (reward +10), 
    action 1 stays in state 0 (reward -1). State 1 is terminal.
    """
    def __init__(self):
        self.state = 0

    def reset(self) -> int:
        """Reset the environment to initial state."""
        self.state = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take a single step in the environment.
        
        Args:
            action: Integer action (0 or 1)
            
        Returns:
            next_state: New state after taking action
            reward: Reward for this step
            done: Whether episode is terminated
        """
        if action == 0:
            next_state = 1
            reward = 10.0
        else:
            next_state = self.state
            reward = -1.0
        
        done = (next_state == 1)
        return next_state, reward, done

class ActorCritic:
    """Actor-Critic agent for tabular Q-learning.
    
    Uses a critic to store state-action Q-values and an actor that selects actions greedily.
    """
    def __init__(self):
        self.critic: dict[tuple[int, int], float] = {}
        self.gamma: float = 0.9
        self.alpha: float = 0.1

    def select_action(self, state: int) -> int:
        """Select the best action based on current Q-values.
        
        Args:
            state: Current state
            
        Returns:
            Action (0 or 1)
        """
        actions = [0, 1]
        q_values = [
            self.critic.get((state, a), 0.0) 
            for a in actions
        ]
        return 0 if q_values[0] >= q_values[1] else 1

    def update(self, state: int, action: int, next_state: int, reward: float):
        """Update the Q-value for the current state-action pair using TD(0).
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state after taking action
            reward: Reward received
        """
        current_q = self.critic.get((state, action), 0.0)
        
        # Compute max Q-value for next state (terminal states have 0)
        if next_state == 1:
            max_next_q = 0.0
        else:
            actions = [0, 1]
            q_values = [
                self.critic.get((next_state, a), 0.0) 
                for a in actions
            ]
            max_next_q = max(q_values)
        
        td_error = reward + self.gamma * max_next_q - current_q
        new_q = current_q + self.alpha * td_error
        self.critic[(state, action)] = new_q

def train_agent(env: SimpleEnv, actor_critic: ActorCritic, episodes: int = 200) -> list[float]:
    """Train the agent for a specified number of episodes.
    
    Args:
        env: Environment instance
        actor_critic: Agent instance
        episodes: Number of training episodes
        
    Returns:
        List of total rewards per episode
    """
    rewards_per_episode = []
    
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action = actor_critic.select_action(state)
            next_state, reward, done = env.step(action)
            actor_critic.update(state, action, next_state, reward)
            
            total_reward += reward
            state = next_state
            
        rewards_per_episode.append(total_reward)
    
    return rewards_per_episode

def evaluate_agent(env: SimpleEnv, actor_critic: ActorCritic, episodes: int = 20) -> list[float]:
    """Evaluate the agent after training.
    
    Args:
        env: Environment instance
        actor_critic: Agent instance
        episodes: Number of evaluation episodes
        
    Returns:
        List of total rewards per episode
    """
    rewards_per_episode = []
    
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action = actor_critic.select_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            
        rewards_per_episode.append(total_reward)
    
    return rewards_per_episode

def generate_csv_report(rewards_per_episode: list[float], evaluation_rewards: list[float]):
    """Generate a CSV report of training and evaluation metrics.
    
    Args:
        rewards_per_episode: Training rewards per episode
        evaluation_rewards: Evaluation rewards per episode
    """
    filename = "generation_report.csv"
    
    # Calculate average rewards
    avg_train_reward = sum(rewards_per_episode) / len(rewards_per_episode)
    avg_eval_reward = sum(evaluation_rewards) / len(evaluation_rewards)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Training_Reward', 'Evaluation_Reward'])
        
        # Write training rewards
        for i, reward in enumerate(rewards_per_episode):
            writer.writerow([i + 1, round(reward, 2), 'N/A'])
            
        # Write evaluation metrics
        writer.writerow(['Avg_Training', f'{avg_train_reward:.2f}', f'{avg_eval_reward:.2f}'])

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    
    env = SimpleEnv()
    actor_critic = ActorCritic()
    
    # Train agent
    rewards_per_episode = train_agent(env, actor_critic, episodes=200)
    
    # Evaluate agent
    evaluation_rewards = evaluate_agent(env, actor_critic, episodes=20)
    
    # Generate CSV report
    generate_csv_report(rewards_per_episode, evaluation_rewards)
    
    # Print training progress and results
    avg_train_reward = sum(rewards_per_episode) / len(rewards_per_episode)
    print(f"Training completed. Average reward per episode: {avg_train_reward:.2f}")
    print(f"Evaluation average reward: {sum(evaluation_rewards)/len(evaluation_rewards):.2f}")
