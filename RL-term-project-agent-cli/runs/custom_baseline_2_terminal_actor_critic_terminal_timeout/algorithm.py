import random
import csv
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

def step(state: int, action: int) -> Tuple[int, int, bool]:
    """Take a step in the environment."""
    if action == 0:  # left
        next_state = max(0, state - 1)
    else:  # right
        next_state = min(9, state + 1)
    reward = 10 if next_state == 9 else -1
    done = next_state == 9
    return next_state, reward, done

if __name__ == '__main__':
    random.seed(42)
    
    # Initialize actor and critic
    actor: Dict[int, List[float]] = {s: [0.0, 0.0] for s in range(10)}
    critic: Dict[int, float] = {s: 0.0 for s in range(10)}
    
    gamma = 0.9
    alpha = 0.1
    training_episodes = 80
    evaluation_episodes = 10
    
    results = []
    
    # Train for 80 episodes
    for episode in range(training_episodes):
        state = 0
        steps = 0
        total_reward = 0
        done = False
        termination_reason = "timeout"  # default
        while not done and steps < 30:
            # Compute probabilities via softmax
            logits = actor[state]
            sum_exp = sum(math.exp(x) for x in logits)
            probs = [math.exp(x) / sum_exp for x in logits]
            
            # Select action using current policy
            action = random.choices([0, 1], weights=probs, k=1)[0]
            
            next_state, reward, done_flag = step(state, action)
            steps += 1
            total_reward += reward
            
            # Check termination conditions
            if next_state == 9:
                done = True
                termination_reason = "goal"
            elif steps == 30:
                done = True
                termination_reason = "timeout"
            else:
                # Compute TD error
                td_error = reward + gamma * critic[next_state] - critic[state]
                # Update critic
                critic[state] += alpha * td_error
                # Update actor: add TD error to the action's logit
                actor[state][action] += alpha * td_error
        
        results.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'termination': termination_reason
        })
    
    # Generate CSV report
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'total_reward', 'steps', 'termination'])
        writer.writeheader()
        writer.writerows(results)
    
    # Textual policy visualization
    print("Policy visualization:")
    for s in range(10):
        logits = actor[s]
        sum_exp = sum(math.exp(x) for x in logits)
        probs = [math.exp(x) / sum_exp for x in logits]
        print(f"State {s}: left={probs[0]:.2f}, right={probs[1]:.2f}")
