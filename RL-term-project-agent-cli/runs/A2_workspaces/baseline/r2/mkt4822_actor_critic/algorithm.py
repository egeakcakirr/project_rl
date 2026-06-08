from __future__ import annotations
import random
import csv

class GridWorld:
    """Simple 1D grid environment where agent moves towards target position (5)."""

    def __init__(self):
        self.position = 0
        self.target = 5

    def reset(self) -> int:
        """Reset the environment to initial state."""
        self.position = 0
        return self.position

    def step(self, action: int) -> tuple[int, float, bool]:
        """
        Take an action in the environment.

        Args:
            action: Integer action (0 for left, 1 for right).

        Returns:
            next_state: Current position after action.
            reward: Immediate reward for this transition.
            done: Whether the episode ended.
        """
        if action == 0:
            self.position -= 1
        else:
            self.position += 1

        reward = 1.0 if self.position >= self.target else -0.1
        done = self.position >= self.target
        return self.position, reward, done


def choose_action(state: int, value_function: dict[int, float]) -> int:
    """
    Select an action based on the current state and value function.

    Args:
        state: Current position.
        value_function: Dictionary mapping states to estimated values.

    Returns:
        Action (0 or 1).
    """
    return random.choice([0, 1])


def train() -> list[float]:
    """Train the agent for a fixed number of episodes using TD learning."""
    random.seed(42)
    env = GridWorld()
    value_function: dict[int, float] = {0: 0.0}
    gamma = 0.95
    alpha = 0.1
    episodes = 20

    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = choose_action(state, value_function)
            next_state, reward, done = env.step(action)
            
            # Compute TD error
            if done:
                td_error = reward - value_function.get(next_state, 0.0)
            else:
                td_error = reward + gamma * value_function.get(next_state, 0.0) - value_function.get(state, 0.0)
                
            # Update current state's value in the function
            value_function[state] = value_function.get(state, 0.0) + alpha * td_error
            
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        rewards.append(total_reward)
    
    return rewards


def evaluate() -> float:
    """Evaluate the agent's performance after training."""
    random.seed(42)
    env = GridWorld()
    value_function: dict[int, float] = {0: 0.0}
    gamma = 0.95
    alpha = 0.1
    
    total_reward = 0
    for _ in range(3):  # Run 3 episodes for evaluation
        state = env.reset()
        while True:
            action = random.choice([0, 1])
            next_state, reward, done = env.step(action)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
                
    return total_reward


def generate_csv_report(rewards: list[float]):
    """Generate CSV report with training rewards."""
    filename = 'generation_report.csv'
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward'])
        for episode_idx, reward in enumerate(rewards):
            writer.writerow([episode_idx + 1, reward])


if __name__ == "__main__":
    print("Starting training...")
    rewards = train()
    
    print("\nTraining completed.")
    avg_reward = sum(rewards) / len(rewards)
    print(f"Average training reward: {avg_reward:.2f}")
    
    print("\nEvaluation:")
    eval_reward = evaluate()
    print(f"Evaluation reward: {eval_reward:.2f}")
    
    generate_csv_report(rewards)
    print("Report generated as generation_report.csv")
