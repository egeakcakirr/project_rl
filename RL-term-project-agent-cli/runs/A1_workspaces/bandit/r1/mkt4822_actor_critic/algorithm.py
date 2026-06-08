from typing import List, Dict, Tuple, Optional
import random
import time
import csv

class GridEnvironment:
    """Simple grid environment for demonstration purposes.
    
    The agent moves in a 1D grid from state 0 to 9. The target is at position 5.
    Actions: 0 (left), 1 (right).
    """
    def __init__(self, target: int = 5):
        self.target = target
        self.gamma = 0.9

    def reset(self) -> int:
        """Reset environment to initial state."""
        self.state = 0
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """Take a single step in the environment.
        
        Args:
            action: Action (0 for left, 1 for right)
            
        Returns:
            next_state: New state after action
            reward: Reward received
            done: Whether episode ended
        """
        if action == 0:
            next_state = max(0, self.state - 1)
        else:
            next_state = min(9, self.state + 1)
        
        reward = 1.0 if next_state == self.target else 0.0
        done = (next_state == self.target)
        self.state = next_state
        return next_state, reward, done

class ActorCritic:
    """Actor-Critic reinforcement learning agent for tabular environments.
    
    This implementation uses a simple policy and value function update.
    """
    def __init__(self, env: GridEnvironment):
        self.env = env
        # Policy: [prob_left, prob_right] for each state (0-9)
        self.policy = [[0.5, 0.5] for _ in range(10)]
        # Value function V[s]
        self.V = [0.0] * 10

    def select_action(self, state: int) -> int:
        """Select action based on current policy.
        
        Args:
            state: Current state (0-9)
            
        Returns:
            Action (0 or 1)
        """
        if random.random() < self.policy[state][0]:
            return 0
        else:
            return 1

    def update_critic(self, state: int, next_state: int, reward: float, done: bool):
        """Update value function using TD error.
        
        Args:
            state: Current state
            next_state: Next state
            reward: Reward received
            done: Episode termination flag
        """
        gamma = self.env.gamma
        td_error = reward + gamma * self.V[next_state] - self.V[state]
        self.V[state] += 0.1 * td_error

    def update_policy(self, state: int, action: int, next_state: int, reward: float, done: bool):
        """Update policy probabilities.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Episode termination flag
        """
        if action == 0:
            # Increase probability of left action (but not beyond 1.0)
            self.policy[state][0] = min(1.0, max(0.0, self.policy[state][0] + 0.01))
        else:
            self.policy[state][1] = min(1.0, max(0.0, self.policy[state][1] + 0.01))

def train_actor_critic(
    env: GridEnvironment,
    actor_critic: ActorCritic,
    episodes: int = 100,
    max_steps_per_episode: int = 50
) -> List[Tuple[int, float, int]]:
    """Train the actor-critic agent.
    
    Args:
        env: Environment instance
        actor_critic: Agent to train
        episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        
    Returns:
        List of tuples (episode, total_reward, steps)
    """
    episode_data = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done and steps < max_steps_per_episode:
            action = actor_critic.select_action(state)
            next_state, reward, done = env.step(action)
            actor_critic.update_critic(state, next_state, reward, done)
            actor_critic.update_policy(state, action, next_state, reward, done)
            total_reward += reward
            state = next_state
            steps += 1
        episode_data.append((episode, total_reward, steps))
    return episode_data

def evaluate_actor_critic(
    actor_critic: ActorCritic,
    env: GridEnvironment,
    episodes: int = 5
) -> float:
    """Evaluate the trained agent.
    
    Args:
        actor_critic: Trained agent
        env: Environment instance
        episodes: Number of evaluation episodes
        
    Returns:
        Average reward across episodes
    """
    total_rewards = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = actor_critic.select_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
    return sum(total_rewards) / len(total_rewards)

def generate_csv_report(episode_data: List[Tuple[int, float, int]], avg_reward: float):
    """Generate CSV report with training metrics.
    
    Args:
        episode_data: Training data per episode
        avg_reward: Average reward from evaluation
    """
    csv_path = "generation_report.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['episode', 'reward', 'steps'])
        for episode, total_reward, steps in episode_data:
            writer.writerow([episode, total_reward, steps])
        # Add evaluation metrics
        writer.writerow(['evaluation_avg_reward', avg_reward])

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    
    # Initialize environment and agent
    env = GridEnvironment(target=5)
    actor_critic = ActorCritic(env)
    
    print("Starting training...")
    start_time = time.time()
    episode_data = train_actor_critic(
        env,
        actor_critic,
        episodes=5,  # Reduced for faster execution
        max_steps_per_episode=10  # Reduced for faster execution
    )
    training_duration = time.time() - start_time
    
    # Evaluate the agent
    avg_reward_eval = evaluate_actor_critic(actor_critic, env)
    
    print(f"\nTraining completed in {training_duration:.2f} seconds")
    print(f"Average evaluation reward: {avg_reward_eval:.2f}")
    
    # Generate CSV report
    generate_csv_report(episode_data, avg_reward_eval)
    
    print("\nCSV report generated at 'generation_report.csv'")
