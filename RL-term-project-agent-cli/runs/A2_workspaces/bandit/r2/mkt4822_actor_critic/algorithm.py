"""Actor-Critic reinforcement learning example with standard library only."""

from __future__ import annotations

import random
import csv
from typing import List, Dict, Tuple


class SimpleEnv:
    """A minimal 2D grid environment for testing Actor-Critic.

    The agent starts at (0, 0) and aims to reach (2, 2).
    Actions: 0=right, 1=down, 2=left, 3=up.
    """

    def __init__(self):
        self.state = (0, 0)
        self.goal = (2, 2)

    def reset(self) -> Tuple[int, int]:
        """Reset environment to initial state."""
        self.state = (0, 0)
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Take one step in the environment.

        Args:
            action: Integer action (0=right, 1=down, 2=left, 3=up)

        Returns:
            next_state: New state tuple
            reward: Scalar reward
            done: Boolean indicating terminal state
        """
        x, y = self.state
        if action == 0:
            new_x = x + 1
            new_y = y
        elif action == 1:
            new_x = x
            new_y = y + 1
        else:
            new_x, new_y = x, y

        # Check if goal reached
        if (new_x, new_y) == self.goal:
            reward = 10.0
            done = True
        elif new_x < 0 or new_x > 2 or new_y < 0 or new_y > 2:
            reward = -1.0
            done = True
        else:
            reward = -0.1
            done = False

        self.state = (new_x, new_y)
        return self.state, reward, done


class ActorCritic:
    """Actor-Critic agent for simple grid environment."""

    def __init__(self, env: SimpleEnv):
        self.env = env
        self.policy: Dict[Tuple[int, int], List[float]] = {}
        self.critic_values: Dict[Tuple[int, int], float] = {}

    def reset_policy(self) -> None:
        """Initialize policy and critic values for all states."""
        initial_states = [(x, y) for x in range(3) for y in range(3)]
        for state in initial_states:
            self.policy[state] = [0.25, 0.25, 0.25, 0.25]
            self.critic_values[state] = 0.0

    def select_action(self, state: Tuple[int, int]) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state (x, y)

        Returns:
            Action index (0=right, 1=down, 2=left, 3=up)
        """
        epsilon = 0.1
        if random.random() < epsilon:
            return random.randint(0, 3)
        # Find best action based on critic values
        max_value = -float('inf')
        best_action = None
        for action in range(4):
            if self.critic_values[state] > max_value:
                max_value = self.critic_values[state]
                best_action = action
        return best_action

    def update_critic(self, state: Tuple[int, int], next_state: Tuple[int, int], reward: float, done: bool) -> None:
        """Update critic values using TD(0) error.

        Args:
            state: Current state
            next_state: Next state
            reward: Reward received
            done: Terminal state flag
        """
        gamma = 0.95
        if done:
            td_error = reward - self.critic_values[state]
        else:
            td_error = reward + gamma * self.critic_values[next_state] - self.critic_values[state]
        self.critic_values[state] += 0.1 * td_error

    def update_policy(self, state: Tuple[int, int], action: int, next_state: Tuple[int, int], reward: float, done: bool) -> None:
        """Update policy probabilities based on critic values.

        This is a simplified version for demonstration.
        """
        pass


if __name__ == "__main__":
    random.seed(42)
    env = SimpleEnv()
    ac = ActorCritic(env)
    ac.reset_policy()

    episodes = 10
    training_data = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = ac.select_action(state)
            next_state, reward, done = env.step(action)
            ac.update_critic(state, next_state, reward, done)
            steps += 1
            total_reward += reward
            state = next_state

        training_data.append({
            'Episode': episode + 1,
            'Steps': steps,
            'Total Reward': total_reward
        })

    # Evaluate after training
    state = env.reset()
    total_eval_reward = 0.0
    eval_steps = 0
    done = False
    while not done:
        action = ac.select_action(state)
        next_state, reward, done = env.step(action)
        total_eval_reward += reward
        eval_steps += 1
        state = next_state

    # Generate CSV report
    with open('generation_report.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Steps', 'Total Reward'])
        for data in training_data:
            writer.writerow([data['Episode'], data['Steps'], data['Total Reward']])
        writer.writerow(['Evaluation', eval_steps, total_eval_reward])

    print("Training completed. Evaluation results:")
    print(f"Steps: {eval_steps}, Total Reward: {total_eval_reward:.2f}")
