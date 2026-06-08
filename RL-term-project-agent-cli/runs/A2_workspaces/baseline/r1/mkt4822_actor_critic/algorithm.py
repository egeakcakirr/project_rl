"""Actor-Critic reinforcement learning example using standard library only."""

from typing import Dict, List, Tuple, Optional
import random
import csv
import os

class RandomWalkEnv:
    """Simple 1D walk environment with states 0-5 (target state 5)."""
    
    def __init__(self, start_state: int = 1):
        self.start_state = start_state
        self.current_state = start_state
    
    def reset(self) -> int:
        """Reset the environment to initial state."""
        self.current_state = self.start_state
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take an action and return next state, reward, done flag.
        
        Args:
            action: 0 for left, 1 for right
        
        Returns:
            next_state (int): New state after action
            reward (float): Reward for this step
            done (bool): Whether episode ended
        """
        if action == 0:
            new_state = max(0, self.current_state - 1)
        else:
            new_state = min(5, self.current_state + 1)
        
        reward = 1.0 if new_state == 5 else 0.0
        done = (new_state == 5)
        self.current_state = new_state
        return new_state, reward, done

class ActorCritic:
    """Simple tabular Actor-Critic agent for discrete actions."""
    
    def __init__(self):
        # Policy: action for each state (0 or 1)
        self.policy = {i: 0 for i in range(6)}  # Default to action 0
        # Critic values (V(s))
        self.critic_values = {i: 0.0 for i in range(6)}
    
    def act(self, state: int) -> int:
        """Select action based on current policy."""
        return self.policy[state]
    
    def update_critic(self, prev_state: int, next_state: int, reward: float, done: bool):
        """
        Update critic values using TD(0) error.
        
        Args:
            prev_state (int): State before action
            next_state (int): State after action
            reward (float): Immediate reward
            done (bool): Episode termination flag
        """
        if not done:
            td_error = reward + 0.9 * self.critic_values[next_state] - self.critic_values[prev_state]
            self.critic_values[prev_state] += 0.5 * td_error

def train_actor_critic(env: RandomWalkEnv, ac: ActorCritic, episodes: int = 5):
    """Train the actor-critic agent for a number of episodes with step limits."""
    results = []
    
    max_steps_per_episode = 20
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done and steps < max_steps_per_episode:
            action = ac.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            # Update critic with previous state (current state before step)
            ac.update_critic(prev_state=state, next_state=next_state, reward=reward, done=done)
            
            state = next_state
        
        results.append((episode + 1, steps, total_reward))
    
    return results

def evaluate_agent(env: RandomWalkEnv, ac: ActorCritic) -> Tuple[int, float]:
    """Evaluate the agent on a single test run with step limit."""
    state = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    max_steps = 20
    
    while not done and steps < max_steps:
        action = ac.act(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        state = next_state
    
    return steps, total_reward

def main():
    """Main execution block for training and evaluation."""
    random.seed(42)  # Deterministic behavior
    
    env = RandomWalkEnv(start_state=1)
    ac = ActorCritic()
    
    # Train the agent with step limit to prevent infinite loops
    results = train_actor_critic(env, ac, episodes=5)
    
    # Evaluate after training
    steps_eval, reward_eval = evaluate_agent(env, ac)
    
    # Generate CSV report
    csv_path = 'generation_report.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'steps', 'total_reward'])
        for episode, steps, total_reward in results:
            writer.writerow([episode, steps, total_reward])
    
    # Print summary
    print(f"Training completed. Evaluation: {steps_eval} steps, {reward_eval:.2f} reward")
    print(f"Report saved to {csv_path}")

if __name__ == "__main__":
    main()
