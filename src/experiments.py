# %%
import os
import matplotlib.pyplot as plt
from agent import DQNAgent
import numpy as np
import random
from utils import get_overlap
def experiment1():
    random.seed(0)
    dataset = random.sample(get_overlap(), 90)
    burn_in = dataset[:30]
    train = dataset[30:60]
    val = dataset[60:90]

    for i in range(1):
        print(f"Starting run {i}")
        agent = DQNAgent(burn_in=burn_in, train=train, val=val)
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
