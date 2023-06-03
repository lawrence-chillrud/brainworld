# %%
import os
import matplotlib.pyplot as plt
from agent import DQNAgent
import numpy as np

def experiment1():
    for i in range(3):
        print(f"Starting run {i}")
        agent = DQNAgent()
        scores, found_lesion = agent.training()
        plt.plot(np.arange(agent.EPISODES), scores, label = f"Run {i + 1}")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.title("Experiment 1: Reward vs Episode")
    plt.savefig(f'/home/{os.getlogin()}/brainworld/output/figures/experiment1.png') 
    # plt.show()
    # agent.test()

if __name__ == "__main__":
    experiment1()

# %%
