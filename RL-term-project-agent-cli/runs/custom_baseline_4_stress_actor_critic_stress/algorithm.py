from typing import List, Dict, Tuple, Optional, Any
import random
import time
import csv
import math

random.seed(42)

MAX_STATE = 14
GOAL_STATE = 14

def step_environment(state: int, action: int) -> Tuple[int, int]:
    """Step the environment with given action."""
    if action == 0:  # left
        next_state = state - 1
    else:  # right
        next_state = state + 1
    next_state = max(0, min(MAX_STATE, next_state))
    reward = 15 if next_state == GOAL_STATE else -1
    return next_state, reward

def initialize_actor_critic() -> Tuple[List[List[float]], List[float]]:
    """Initialize tabular actor and critic."""
    theta = [[0.0, 0.0] for _ in range(MAX_STATE + 1)]  # 0 to 14
    Q = [0.0] * (MAX_STATE + 1)
    return theta, Q

def select_action(state: int, theta: List[List[float]]) -> int:
    """Select action using softmax policy."""
    actions = []
    action_probs = []
    if state > 0:
        actions.append(0)
        action_probs.append(theta[state][0])
    if state < MAX_STATE:
        actions.append(1)
        action_probs.append(theta[state][1])
    
    exp_vals = [math.exp(p) for p in action_probs]
    total = sum(exp_vals)
    probs = [exp_val / total for exp_val in exp_vals]
    
    return random.choices(actions, weights=probs, k=1)[0]

def update_critic(Q: List[float], state: int, next_state: int, reward: int, gamma: float, alpha: float) -> None:
    """Update critic Q values using TD error."""
    td_error = reward + gamma * Q[next_state] - Q[state]
    Q[state] += alpha * td_error

def update_actor(theta: List[List[float]], Q: List[float], state: int, action: int, gamma: float, alpha: float) -> None:
    """Update actor parameters using policy gradient."""
    actions = []
    if state > 0:
        actions.append(0)
    if state < MAX_STATE:
        actions.append(1)
    
    theta_vals = [theta[state][i] for i in actions]
    exp_vals = [math.exp(p) for p in theta_vals]
    total = sum(exp_vals)
    probs = [exp_val / total for exp_val in exp_vals]
    
    action_index = actions.index(action)
    
    if action_index == 0:
        theta[state][0] += alpha * (probs[0] - 1) * (Q[state] - Q[state - 1])
    else:
        theta[state][1] += alpha * (probs[1] - 1) * (Q[state] - Q[state + 1])

if __name__ == '__main__':
    gamma = 0.9
    alpha = 0.1
    num_episodes = 500
    eval_episodes = 20

    theta, Q = initialize_actor_critic()
    report = []

    print("Training for 500 episodes...")
    for episode in range(num_episodes):
        state = 0
        steps = 0
        total_reward = 0
        terminated = False

        while steps < 70 and not terminated:
            action = select_action(state, theta)
            next_state, reward = step_environment(state, action)
            update_critic(Q, state, next_state, reward, gamma, alpha)
            update_actor(theta, Q, state, action, gamma, alpha)
            state = next_state
            steps += 1
            total_reward += reward
            if next_state == GOAL_STATE:
                terminated = True

        report.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'termination': 1 if terminated else 0
        })

        print(f"Episode {episode}: Total reward = {total_reward}, Steps = {steps}, Termination = {1 if terminated else 0}")

    # Generate CSV report
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'total_reward', 'steps', 'termination'])
        writer.writeheader()
        writer.writerows(report)

    # Evaluation loop
    print("\nEvaluating for 20 episodes...")
    eval_report = []
    for _ in range(eval_episodes):
        state = 0
        steps = 0
        total_reward = 0
        terminated = False
        while steps < 70 and not terminated:
            if state > 0 and state < MAX_STATE:
                left_q = Q[state - 1]
                right_q = Q[state + 1]
                action = 0 if left_q > right_q else 1
            elif state > 0:
                action = 0
            else:
                action = 1
            next_state, reward = step_environment(state, action)
            state = next_state
            steps += 1
            total_reward += reward
            if next_state == GOAL_STATE:
                terminated = True
        eval_report.append({
            'episode': _,
            'total_reward': total_reward,
            'steps': steps,
            'termination': 1 if terminated else 0
        })

    print(f"Evaluation summary: {eval_report}")

    # Textual visualization
    print("\nPolicy visualization:")
    for s in range(MAX_STATE + 1):
        policy_str = f"State {s}: "
        if s == 0:
            policy_str += "Right"
        elif s == MAX_STATE:
            policy_str += "Left"
        else:
            left_prob = math.exp(theta[s][0]) / (math.exp(theta[s][0]) + math.exp(theta[s][1]))
            policy_str += f"Left ({left_prob:.2f}) | Right ({1 - left_prob:.2f})"
        print(policy_str)

    print("\nQ-values visualization:")
    for s in range(MAX_STATE + 1):
        print(f"State {s}: {Q[s]:.2f}")
