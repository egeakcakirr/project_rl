import random
from typing import List, Dict

class SimpleEnv:
    """Simple deterministic environment for demonstration with clamped states."""
    
    def __init__(self):
        self.state = 0
    
    def reset(self) -> int:
        """Reset the environment to initial state (state 0)."""
        self.state = 0
        return self.state
    
    def step(self, action: int) -> tuple[int, float, bool]:
        """
        Execute an action in the environment with clamped states.
        
        Args:
            action: Integer action (0 or 1)
            
        Returns:
            next_state: Current state after action (clamped between 0 and 2)
            reward: Reward for this transition
            done: Whether episode terminated
        """
        if action == 0:
            self.state = min(2, self.state + 1)
        else:
            self.state = max(0, self.state - 1)
        
        # Goal state is 2 with a positive reward; otherwise negative reward
        done = (self.state == 2)
        reward = 1.0 if done else -0.1
        
        return self.state, reward, done

class ActorCritic:
    """Actor-Critic agent for simple environment."""
    
    def __init__(self, env: SimpleEnv):
        self.env = env
        # Initialize policy as uniform over actions for each state (0-2)
        self.policy = {state: [0.5, 0.5] for state in range(3)}
        self.value = {state: 0.0 for state in range(3)}
    
    def select_action(self, state: int) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < 0.1:  # Exploration
            return random.randint(0, 1)
        else:
            probs = self.policy[state]
            total = sum(probs)
            normalized_probs = [p / total for p in probs]
            return random.choices([0, 1], weights=normalized_probs)[0]
    
    def update_policy(self, state: int, action: int, reward: float):
        """Update policy based on TD error (simplified)."""
        pass

def train_agent(env: SimpleEnv, num_episodes: int = 100) -> tuple[ActorCritic, List[float]]:
    """
    Train the actor-critic agent for a fixed number of episodes.
    
    Args:
        env: Environment instance
        num_episodes: Number of training episodes
        
    Returns:
        (agent, total_rewards): Trained agent and list of episode rewards
    """
    agent = ActorCritic(env)
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Update value function (critic) using TD error
            td_error = reward + 0.9 * agent.value[next_state] - agent.value[state]
            agent.value[state] += 0.1 * td_error
            
            # Update policy (actor) for the current state
            if random.random() < 0.5:
                agent.policy[state][action] += 0.01
            else:
                agent.policy[state][action] -= 0.01
            
            state = next_state
            total_reward += reward
        
        total_rewards.append(total_reward)
    
    return agent, total_rewards

def evaluate_agent(agent: ActorCritic, env: SimpleEnv) -> float:
    """Evaluate the trained agent in a single episode."""
    state = env.reset()
    total_reward = 0.0
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
    
    return total_reward

def generate_report(agent: ActorCritic, env: SimpleEnv, total_rewards: List[float]):
    """Generate CSV report with training and evaluation metrics."""
    import csv
    with open('generation_report.csv', 'w') as f:
        writer = csv.writer(f)
        # Write headers
        writer.writerow(['Episode', 'Total Reward'])
        
        # Write training rewards
        for i, reward in enumerate(total_rewards):
            writer.writerow([i + 1, round(reward, 2)])
        
        # Add evaluation metrics
        writer.writerow(['Evaluation', 'Final Reward'])
        final_reward = evaluate_agent(agent, env)
        writer.writerow([None, round(final_reward, 2)])

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    
    # Create environment and train agent
    env = SimpleEnv()
    agent, total_rewards = train_agent(env, num_episodes=100)
    
    # Generate report
    generate_report(agent, env, total_rewards)
    
    final_reward = evaluate_agent(agent, env)
    print(f"Training complete. Average reward per episode: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Final evaluation reward: {final_reward:.2f}")
