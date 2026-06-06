import random
import sys
import os
import csv


class SimpleEnv:
    """A simple 1D environment for demonstration of Actor-Critic RL.
    
    The agent moves in a 1D space with position and velocity. Actions: 
        - 0: stop
        - 1: move right (increase velocity)
        - 2: move left (decrease velocity)
    Target position is 5.0 units.
    """
    def __init__(self):
        self.position = 0.0
        self.velocity = 0.0
        self.target = 5.0

    def reset(self) -> list[float]:
        """Reset the environment to initial state."""
        self.position = 0.0
        self.velocity = 0.0
        return [self.position, self.velocity]

    def step(self, action: int) -> tuple[list[float], float, bool]:
        """Take a single step in the environment.
        
        Args:
            action: Action to take (0: stop, 1: right, 2: left)
            
        Returns:
            next_state: [position, velocity]
            reward: float
            done: bool (whether episode ended)
        """
        if action == 1:
            self.velocity += 0.1
        elif action == 2:
            self.velocity -= 0.1

        self.position += self.velocity * 0.1
        done = self.position >= self.target
        reward = 1.0 if done else -0.1
        return [self.position, self.velocity], reward, done


class Critic:
    """Critic for Actor-Critic RL.
    
    Uses a linear value function: V(s) = w0 * position + w1 * velocity.
    """
    def __init__(self):
        self.weights: list[float] = [0.0, 0.0]

    def predict(self, state: list[float]) -> float:
        """Predict the value of a state."""
        return self.weights[0] * state[0] + self.weights[1] * state[1]

    def update(self, state: list[float], reward: float, next_state: list[float], gamma: float = 0.9) -> None:
        """Update critic weights using TD error.
        
        Args:
            state: Current state
            reward: Immediate reward
            next_state: Next state after action
            gamma: Discount factor (default: 0.9)
        """
        current_value = self.predict(state)
        next_value = self.predict(next_state)
        td_error = reward + gamma * next_value - current_value
        # Update weights with gradient descent step
        self.weights[0] += 0.1 * td_error * state[0]
        self.weights[1] += 0.1 * td_error * state[1]


class Actor:
    """Actor for Actor-Critic RL.
    
    Uses epsilon-greedy policy to select actions.
    """
    def __init__(self, epsilon: float = 0.2):
        self.epsilon = epsilon

    def choose_action(self, state: list[float]) -> int:
        """Select an action with exploration."""
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        else:
            # Prefer moving right (action 1) for deterministic behavior
            return 1


def train_actor_critic(env: SimpleEnv, critic: Critic, episodes: int = 500) -> list[float]:
    """Train the Actor-Critic model.
    
    Args:
        env: Environment instance
        critic: Critic instance
        episodes: Number of training episodes
        
    Returns:
        list of total rewards per episode
    """
    gamma = 0.9
    actor = Actor()
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = actor.choose_action(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Update critic with TD error
            critic.update(state, reward, next_state, gamma)
            
            state = next_state
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    return total_rewards


def evaluate(env: SimpleEnv, actor: Actor, num_episodes: int = 10) -> float:
    """Evaluate the trained agent.
    
    Args:
        env: Environment instance
        actor: Actor instance
        num_episodes: Number of evaluation episodes
        
    Returns:
        Average reward per episode
    """
    total_reward = 0.0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = actor.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
    
    return total_reward / num_episodes


def generate_report_csv(total_rewards: list[float], eval_reward: float) -> None:
    """Generate CSV report for training and evaluation.
    
    Args:
        total_rewards: List of rewards per episode during training
        eval_reward: Average reward from evaluation episodes
    """
    with open('generation_report.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Total Reward', 'Eval Avg Reward'])
        for i, reward in enumerate(total_rewards):
            writer.writerow([i + 1, reward, eval_reward])


if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    
    env = SimpleEnv()
    critic = Critic()
    total_rewards = train_actor_critic(env, critic)
    actor = Actor()
    eval_reward = evaluate(env, actor)
    
    generate_report_csv(total_rewards, eval_reward)
    
    print("\nTraining completed. Report generated at generation_report.csv")
