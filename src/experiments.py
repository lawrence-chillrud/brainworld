# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import matplotlib.pyplot as plt
from agent import DQNAgent
from brainworld import BrainWorldEnv
import numpy as np
import random
from utils import get_overlap

def experiment(burn_in, train, val, modality="t1ce", action_mask=True, find_max_lesion=True, scale_rewards=False, positional_encoding=False, episodes=80, eval=10):
    env = BrainWorldEnv(modality=modality, action_mask=action_mask, find_max_lesion=find_max_lesion, scale_rewards=scale_rewards)
    agent = DQNAgent(burn_in=burn_in, train=train, val=val, env=env, episodes=episodes, positional_encoding=positional_encoding)
    return agent.training(evaluate_val_every=eval)

def experiment1(burn_in, train, val, epi=40, ev=10):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    val_episode_nos = np.arange(0, epi + 1, ev)
    val_episode_nos[0] = 1

    # scores, found_lesion, overlap_lesion, overlap_lesion_err, mean_val_scores, val_max_acc, val_overlap_acc, val_overlap_rmse = experiment
    for i, (pos_enc, burn_set) in enumerate(zip([False, True, False, True], [None, None, burn_in, burn_in])):
        print(f"Starting experiment 1; sub experiment {i + 1} of 4...")
        burn_lab = False
        if burn_set: burn_lab = True
        name = f"pos_encoding: {pos_enc}, burn_in: {burn_lab}"
        print(name)
        _, _, _, _, mean_val_scores, val_max_acc, val_overlap_acc, val_overlap_rmse = experiment(burn_in=burn_set, train=train, val=val, positional_encoding=pos_enc, episodes=epi, eval=ev)
        
        ax[0, 0].plot(val_episode_nos, mean_val_scores, label=name)
        # ax[0, 0].scatter(val_episode_nos, mean_val_scores, label=name)
        ax[0, 0].set_title("Mean Return")
        ax[0, 0].set_xlabel("Episode")
        ax[0, 0].set_ylabel("Mean Return")
        ax[0, 0].set_xticks(val_episode_nos)

        ax[0, 1].plot(val_episode_nos, val_max_acc, label=name)
        # ax[0, 1].scatter(val_episode_nos, val_max_acc, label=name)
        ax[0, 1].set_title("Accuracy (using max lesions found)")
        ax[0, 1].set_xlabel("Episode")
        ax[0, 1].set_ylabel("Accuracy (using max lesions found)")
        ax[0, 1].set_xticks(val_episode_nos)

        ax[1, 0].plot(val_episode_nos, val_overlap_acc, label=name)
        # ax[1, 0].scatter(val_episode_nos, val_overlap_acc, label=name)
        ax[1, 0].set_title("Accuracy (using overlap lesions)")
        ax[1, 0].set_xlabel("Episode")
        ax[1, 0].set_ylabel("Accuracy (using overlap lesions)")
        ax[1, 0].set_xticks(val_episode_nos)

        ax[1, 1].plot(val_episode_nos, val_overlap_rmse, label=name)
        # ax[1, 1].scatter(val_episode_nos, val_overlap_rmse, label=name)
        ax[1, 1].set_title("RMSE (of percent overlap)")
        ax[1, 1].set_xlabel("Episode")
        ax[1, 1].set_ylabel("RMSE (of percent overlap)")
        ax[1, 1].set_xticks(val_episode_nos)

    fig.suptitle("Experiment 1 (Action masking): Validation Performance vs Episode")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left')
    fig.tight_layout()
    fig.savefig(f"/home/{os.getlogin()}/brainworld/output/figures/experiment1.png")

def experiment2(burn_in, train, val, epi=40, ev=10):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    val_episode_nos = np.arange(0, epi + 1, ev)
    val_episode_nos[0] = 1

    # scores, found_lesion, overlap_lesion, overlap_lesion_err, mean_val_scores, val_max_acc, val_overlap_acc, val_overlap_rmse = experiment
    for i, (pos_enc, burn_set) in enumerate(zip([False, True, False, True], [None, None, burn_in, burn_in])):
        print(f"Starting experiment 2; sub experiment {i + 1} of 4...")
        burn_lab = False
        if burn_set: burn_lab = True
        name = f"pos_encoding: {pos_enc}, burn_in: {burn_lab}"
        print(name)
        _, _, _, _, mean_val_scores, val_max_acc, val_overlap_acc, val_overlap_rmse = experiment(burn_in=burn_set, train=train, val=val, action_mask=False, positional_encoding=pos_enc, episodes=epi, eval=ev)
        
        ax[0, 0].plot(val_episode_nos, mean_val_scores, label=name)
        # ax[0, 0].scatter(val_episode_nos, mean_val_scores, label=name)
        ax[0, 0].set_title("Mean Return")
        ax[0, 0].set_xlabel("Episode")
        ax[0, 0].set_ylabel("Mean Return")
        ax[0, 0].set_xticks(val_episode_nos)

        ax[0, 1].plot(val_episode_nos, val_max_acc, label=name)
        # ax[0, 1].scatter(val_episode_nos, val_max_acc, label=name)
        ax[0, 1].set_title("Accuracy (using max lesions found)")
        ax[0, 1].set_xlabel("Episode")
        ax[0, 1].set_ylabel("Accuracy (using max lesions found)")
        ax[0, 1].set_xticks(val_episode_nos)

        ax[1, 0].plot(val_episode_nos, val_overlap_acc, label=name)
        # ax[1, 0].scatter(val_episode_nos, val_overlap_acc, label=name)
        ax[1, 0].set_title("Accuracy (using overlap lesions)")
        ax[1, 0].set_xlabel("Episode")
        ax[1, 0].set_ylabel("Accuracy (using overlap lesions)")
        ax[1, 0].set_xticks(val_episode_nos)

        ax[1, 1].plot(val_episode_nos, val_overlap_rmse, label=name)
        # ax[1, 1].scatter(val_episode_nos, val_overlap_rmse, label=name)
        ax[1, 1].set_title("RMSE (of percent overlap)")
        ax[1, 1].set_xlabel("Episode")
        ax[1, 1].set_ylabel("RMSE (of percent overlap)")
        ax[1, 1].set_xticks(val_episode_nos)

    fig.suptitle("Experiment 2 (Grid wrapping): Validation Performance vs Episode")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left')
    fig.tight_layout()
    fig.savefig(f"/home/{os.getlogin()}/brainworld/output/figures/experiment2.png")

def experiment3(burn_in, train, val, epi=40, ev=10):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    val_episode_nos = np.arange(0, epi + 1, ev)
    val_episode_nos[0] = 1

    # scores, found_lesion, overlap_lesion, overlap_lesion_err, mean_val_scores, val_max_acc, val_overlap_acc, val_overlap_rmse = experiment
    for i, (pos_enc, burn_set) in enumerate(zip([False, True, False, True], [None, None, burn_in, burn_in])):
        print(f"Starting experiment 3; sub experiment {i + 1} of 4...")
        burn_lab = False
        if burn_set: burn_lab = True
        name = f"pos_encoding: {pos_enc}, burn_in: {burn_lab}"
        print(name)
        _, _, _, _, mean_val_scores, val_max_acc, val_overlap_acc, val_overlap_rmse = experiment(burn_in=burn_set, train=train, val=val, action_mask=False, find_max_lesion=False, positional_encoding=pos_enc, episodes=epi, eval=ev)
        
        ax[0, 0].plot(val_episode_nos, mean_val_scores, label=name)
        # ax[0, 0].scatter(val_episode_nos, mean_val_scores, label=name)
        ax[0, 0].set_title("Mean Return")
        ax[0, 0].set_xlabel("Episode")
        ax[0, 0].set_ylabel("Mean Return")
        ax[0, 0].set_xticks(val_episode_nos)

        ax[0, 1].plot(val_episode_nos, val_max_acc, label=name)
        # ax[0, 1].scatter(val_episode_nos, val_max_acc, label=name)
        ax[0, 1].set_title("Accuracy (using max lesions found)")
        ax[0, 1].set_xlabel("Episode")
        ax[0, 1].set_ylabel("Accuracy (using max lesions found)")
        ax[0, 1].set_xticks(val_episode_nos)

        ax[1, 0].plot(val_episode_nos, val_overlap_acc, label=name)
        # ax[1, 0].scatter(val_episode_nos, val_overlap_acc, label=name)
        ax[1, 0].set_title("Accuracy (using overlap lesions)")
        ax[1, 0].set_xlabel("Episode")
        ax[1, 0].set_ylabel("Accuracy (using overlap lesions)")
        ax[1, 0].set_xticks(val_episode_nos)

        ax[1, 1].plot(val_episode_nos, val_overlap_rmse, label=name)
        # ax[1, 1].scatter(val_episode_nos, val_overlap_rmse, label=name)
        ax[1, 1].set_title("RMSE (of percent overlap)")
        ax[1, 1].set_xlabel("Episode")
        ax[1, 1].set_ylabel("RMSE (of percent overlap)")
        ax[1, 1].set_xticks(val_episode_nos)

    fig.suptitle("Experiment 3 (Rewards given with any overlap): Validation Performance vs Episode")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left')
    fig.tight_layout()
    fig.savefig(f"/home/{os.getlogin()}/brainworld/output/figures/experiment3.png")

def experiment4(burn_in, train, val, epi=40, ev=10):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    val_episode_nos = np.arange(0, epi + 1, ev)
    val_episode_nos[0] = 1

    # scores, found_lesion, overlap_lesion, overlap_lesion_err, mean_val_scores, val_max_acc, val_overlap_acc, val_overlap_rmse = experiment
    for i, (pos_enc, burn_set) in enumerate(zip([False, True, False, True], [None, None, burn_in, burn_in])):
        print(f"Starting experiment 4; sub experiment {i + 1} of 4...")
        burn_lab = False
        if burn_set: burn_lab = True
        name = f"pos_encoding: {pos_enc}, burn_in: {burn_lab}"
        print(name)
        _, _, _, _, mean_val_scores, val_max_acc, val_overlap_acc, val_overlap_rmse = experiment(burn_in=burn_set, train=train, val=val, action_mask=False, find_max_lesion=False, scale_rewards=True, positional_encoding=pos_enc, episodes=epi, eval=ev)
        
        ax[0, 0].plot(val_episode_nos, mean_val_scores, label=name)
        # ax[0, 0].scatter(val_episode_nos, mean_val_scores, label=name)
        ax[0, 0].set_title("Mean Return")
        ax[0, 0].set_xlabel("Episode")
        ax[0, 0].set_ylabel("Mean Return")
        ax[0, 0].set_xticks(val_episode_nos)

        ax[0, 1].plot(val_episode_nos, val_max_acc, label=name)
        # ax[0, 1].scatter(val_episode_nos, val_max_acc, label=name)
        ax[0, 1].set_title("Accuracy (using max lesions found)")
        ax[0, 1].set_xlabel("Episode")
        ax[0, 1].set_ylabel("Accuracy (using max lesions found)")
        ax[0, 1].set_xticks(val_episode_nos)

        ax[1, 0].plot(val_episode_nos, val_overlap_acc, label=name)
        # ax[1, 0].scatter(val_episode_nos, val_overlap_acc, label=name)
        ax[1, 0].set_title("Accuracy (using overlap lesions)")
        ax[1, 0].set_xlabel("Episode")
        ax[1, 0].set_ylabel("Accuracy (using overlap lesions)")
        ax[1, 0].set_xticks(val_episode_nos)

        ax[1, 1].plot(val_episode_nos, val_overlap_rmse, label=name)
        # ax[1, 1].scatter(val_episode_nos, val_overlap_rmse, label=name)
        ax[1, 1].set_title("RMSE (of percent overlap)")
        ax[1, 1].set_xlabel("Episode")
        ax[1, 1].set_ylabel("RMSE (of percent overlap)")
        ax[1, 1].set_xticks(val_episode_nos)

    fig.suptitle("Experiment 4 (Scaled rewards given with any overlap): Validation Performance vs Episode")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left')
    fig.tight_layout()
    fig.savefig(f"/home/{os.getlogin()}/brainworld/output/figures/experiment4.png")

def experiment5(burn_in, train, val, epi=40, ev=10):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    val_episode_nos = np.arange(0, epi + 1, ev)
    val_episode_nos[0] = 1

    # scores, found_lesion, overlap_lesion, overlap_lesion_err, mean_val_scores, val_max_acc, val_overlap_acc, val_overlap_rmse = experiment
    for i, (pos_enc, burn_set) in enumerate(zip([False, True, False, True], [None, None, burn_in, burn_in])):
        print(f"Starting experiment 5; sub experiment {i + 1} of 4...")
        burn_lab = False
        if burn_set: burn_lab = True
        name = f"pos_encoding: {pos_enc}, burn_in: {burn_lab}"
        print(name)
        _, _, _, _, mean_val_scores, val_max_acc, val_overlap_acc, val_overlap_rmse = experiment(burn_in=burn_set, train=train, val=val, modality="t2", action_mask=False, find_max_lesion=False, scale_rewards=True, positional_encoding=pos_enc, episodes=epi, eval=ev)
        
        ax[0, 0].plot(val_episode_nos, mean_val_scores, label=name)
        # ax[0, 0].scatter(val_episode_nos, mean_val_scores, label=name)
        ax[0, 0].set_title("Mean Return")
        ax[0, 0].set_xlabel("Episode")
        ax[0, 0].set_ylabel("Mean Return")
        ax[0, 0].set_xticks(val_episode_nos)

        ax[0, 1].plot(val_episode_nos, val_max_acc, label=name)
        # ax[0, 1].scatter(val_episode_nos, val_max_acc, label=name)
        ax[0, 1].set_title("Accuracy (using max lesions found)")
        ax[0, 1].set_xlabel("Episode")
        ax[0, 1].set_ylabel("Accuracy (using max lesions found)")
        ax[0, 1].set_xticks(val_episode_nos)

        ax[1, 0].plot(val_episode_nos, val_overlap_acc, label=name)
        # ax[1, 0].scatter(val_episode_nos, val_overlap_acc, label=name)
        ax[1, 0].set_title("Accuracy (using overlap lesions)")
        ax[1, 0].set_xlabel("Episode")
        ax[1, 0].set_ylabel("Accuracy (using overlap lesions)")
        ax[1, 0].set_xticks(val_episode_nos)

        ax[1, 1].plot(val_episode_nos, val_overlap_rmse, label=name)
        # ax[1, 1].scatter(val_episode_nos, val_overlap_rmse, label=name)
        ax[1, 1].set_title("RMSE (of percent overlap)")
        ax[1, 1].set_xlabel("Episode")
        ax[1, 1].set_ylabel("RMSE (of percent overlap)")
        ax[1, 1].set_xticks(val_episode_nos)

    fig.suptitle("Experiment 5 (Scaled rewards given with any overlap, T2 scans): Validation Performance vs Episode")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left')
    fig.tight_layout()
    fig.savefig(f"/home/{os.getlogin()}/brainworld/output/figures/experiment5.png")

def experiment6(burn_in, train, val, epi=40, ev=10):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    val_episode_nos = np.arange(0, epi + 1, ev)
    val_episode_nos[0] = 1

    # scores, found_lesion, overlap_lesion, overlap_lesion_err, mean_val_scores, val_max_acc, val_overlap_acc, val_overlap_rmse = experiment
    for i, (pos_enc, burn_set) in enumerate(zip([False, True, False, True], [None, None, burn_in, burn_in])):
        print(f"Starting experiment 6; sub experiment {i + 1} of 4...")
        burn_lab = False
        if burn_set: burn_lab = True
        name = f"pos_encoding: {pos_enc}, burn_in: {burn_lab}"
        print(name)
        _, _, _, _, mean_val_scores, val_max_acc, val_overlap_acc, val_overlap_rmse = experiment(burn_in=burn_set, train=train, val=val, modality="t2", action_mask=False, find_max_lesion=True, scale_rewards=True, positional_encoding=pos_enc, episodes=epi, eval=ev)
        
        ax[0, 0].plot(val_episode_nos, mean_val_scores, label=name)
        # ax[0, 0].scatter(val_episode_nos, mean_val_scores, label=name)
        ax[0, 0].set_title("Mean Return")
        ax[0, 0].set_xlabel("Episode")
        ax[0, 0].set_ylabel("Mean Return")
        ax[0, 0].set_xticks(val_episode_nos)

        ax[0, 1].plot(val_episode_nos, val_max_acc, label=name)
        # ax[0, 1].scatter(val_episode_nos, val_max_acc, label=name)
        ax[0, 1].set_title("Accuracy (using max lesions found)")
        ax[0, 1].set_xlabel("Episode")
        ax[0, 1].set_ylabel("Accuracy (using max lesions found)")
        ax[0, 1].set_xticks(val_episode_nos)

        ax[1, 0].plot(val_episode_nos, val_overlap_acc, label=name)
        # ax[1, 0].scatter(val_episode_nos, val_overlap_acc, label=name)
        ax[1, 0].set_title("Accuracy (using overlap lesions)")
        ax[1, 0].set_xlabel("Episode")
        ax[1, 0].set_ylabel("Accuracy (using overlap lesions)")
        ax[1, 0].set_xticks(val_episode_nos)

        ax[1, 1].plot(val_episode_nos, val_overlap_rmse, label=name)
        # ax[1, 1].scatter(val_episode_nos, val_overlap_rmse, label=name)
        ax[1, 1].set_title("RMSE (of percent overlap)")
        ax[1, 1].set_xlabel("Episode")
        ax[1, 1].set_ylabel("RMSE (of percent overlap)")
        ax[1, 1].set_xticks(val_episode_nos)

    fig.suptitle("Experiment 6 (Scaled rewards, find max tumor, T2 scans): Validation Performance vs Episode")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left')
    fig.tight_layout()
    fig.savefig(f"/home/{os.getlogin()}/brainworld/output/figures/experiment6.png")

if __name__ == "__main__":
    random.seed(0)
    dataset = random.sample(get_overlap(), 90)
    burn_in = dataset[:30]
    train = dataset[30:60]
    val = dataset[60:90]
    # experiment1(burn_in, train, val, epi=70, ev=10)
    # experiment2(burn_in, train, val, epi=70, ev=10)
    # experiment3(burn_in, train, val, epi=70, ev=10)
    # experiment4(burn_in, train, val, epi=70, ev=10)
    # experiment5(burn_in, train, val, epi=70, ev=10)
    experiment6(burn_in, train, val, epi=70, ev=5)