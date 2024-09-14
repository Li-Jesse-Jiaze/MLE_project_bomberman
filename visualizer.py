import matplotlib.pyplot as plt
import numpy as np
import re

def parse_log_file(file_path):
    scores_data = []
    
    pattern = r"scores: \[(.*?)\]"
    
    with open(file_path, 'r') as log_file:
        for line in log_file:
            match = re.search(pattern, line)
            if match:
                scores_str = match.group(1)
                scores_list = eval(f"[{scores_str}]")
                scores_data.append(scores_list)
    
    return scores_data

def calculate_moving_averages(scores_data, n):
    agents = set()
    for scores in scores_data:
        for agent, _ in scores:
            agents.add(agent)
    
    agents = list(agents)
    
    agent_scores = {agent: [] for agent in agents}
    agent_wins = {agent: [] for agent in agents}
    
    for scores in scores_data:
        max_score = max([score for _, score in scores])
        for agent, score in scores:
            agent_scores[agent].append(score)
            if score == max_score and max_score != 0:
                agent_wins[agent].append(1)  # win
            else:
                agent_wins[agent].append(0)  # loss

    moving_avg_scores = {agent: [] for agent in agents}
    moving_avg_winrates = {agent: [] for agent in agents}
    
    for agent in agents:
        scores = agent_scores[agent]
        wins = agent_wins[agent]
        
        for i in range(1, len(scores) + 1):
            window_scores = scores[max(0, i - n):i]
            window_wins = wins[max(0, i - n):i]
            
            avg_score = np.mean(window_scores)
            avg_winrate = np.mean(window_wins)
            
            moving_avg_scores[agent].append(avg_score)
            moving_avg_winrates[agent].append(avg_winrate)
    
    return moving_avg_scores, moving_avg_winrates

def visualize_moving_averages(scores_data, n):
    moving_avg_scores, moving_avg_winrates = calculate_moving_averages(scores_data, n)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for agent, avg_scores in moving_avg_scores.items():
        plt.plot(range(n, len(avg_scores) + 1), avg_scores[n-1:], label=f"{agent}")
    plt.xlabel("Game")
    plt.ylabel("Average Score")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for agent, avg_winrates in moving_avg_winrates.items():
        plt.plot(range(n, len(avg_winrates) + 1), avg_winrates[n-1:], label=f"{agent}")
    plt.xlabel("Game")
    plt.ylabel("Average Winrate")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()