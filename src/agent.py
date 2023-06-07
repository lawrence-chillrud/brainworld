from datetime import datetime
import random
import numpy as np
from collections import deque
import random
from models import CNN

class DQNAgent:
    def __init__(self, burn_in, train, val, env, episodes=90, positional_encoding=False):
        self.burn_in = burn_in
        self.train = train
        self.val = val
        self.env = env
        self.EPISODES = episodes
        self.positional_encoding = positional_encoding

        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=15000)
        self.gamma = 0.99 # discount rate
        self.epsilon = 0.3 # exploration rate
        self.epsilon_min = 1e-4
        self.epsilon_decay = 1 - 1e-3 # decay rate on right
        self.batch_size = 128
        self.train_start = 200

        self.model = CNN(input_shape=self.env.state[0].shape, action_space = self.action_size, positional_encoding=self.positional_encoding)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    

    def action_mask(self, q_values, position):
        '''
        Implements action masking, so that the agent can't take actions that are not possible given the current position.
        
        Actions
        -------
        0: stay still
        1: move down
        2: move right
        '''
        if position[0] == self.env.grid_size[0] - 1: # we are already at the bottom row of the grid, so we can't move down
            q_values[1] = -np.inf
        if position[1] == self.env.grid_size[1] - 1: # we are already at the rightmost column of the grid, so we can't move right
            q_values[2] = -np.inf
        
        return q_values


    def sample_action_space(self, position):
        '''
        Samples the action space, but removes actions that are not possible given the current position
        '''
        action_space = dict(zip(["stay still", "move down", "move right", "move up", "move left"], np.arange(self.action_size)))
        if position[0] == self.env.grid_size[0] - 1: # we are already at the bottom row of the grid, so we can't move down
            del action_space["move down"]
        if position[1] == self.env.grid_size[1] - 1: # we are already at the rightmost column of the grid, so we can't move right
            del action_space["move right"]
        if "move up" in action_space.keys() and position[0] == 0: # we are already at the top row of the grid, so we can't move up
            del action_space["move up"]
        if "move left" in action_space.keys() and position[1] == 0: # we are already at the leftmost column of the grid, so we can't move left
            del action_space["move left"]
        
        return np.random.choice(list(action_space.values()))


    def act(self, state):
        '''
        Implements the epsilon-greedy policy
        '''
        patch = state[0]
        position = state[1]

        action_picked = "at random"

        if random.random() < self.epsilon:
            # explore
            if self.env.action_mask:
                action = self.sample_action_space(position)
            else: 
                action = self.env.action_space.sample()
        else:
            # exploit
            action_picked = "by model"
            if self.positional_encoding:
                q_values = self.model.predict([np.expand_dims(patch, axis=0), np.expand_dims(position, axis=0)], verbose=0)[0]
            else:
                q_values = self.model.predict(np.expand_dims(patch, axis=0), verbose=0)[0]
            
            if self.env.action_mask: q_values = self.action_mask(q_values, position)
            action = np.argmax(q_values)

        msg = f"\tStep: {self.env.total_steps + 1}/20, Current pos: {position}, Action taken: {action}, Action selected: {action_picked}"

        return action, msg

    
    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state_patch = np.empty(self.batch_size, dtype=object)
        state_position = np.empty(self.batch_size, dtype=object)
        next_state_patch = np.empty(self.batch_size, dtype=object)
        next_state_position = np.empty(self.batch_size, dtype=object)
        action, reward, done = [], [], []

        # assign data into state, next_state, action, reward and done from minibatch
        for i in range(self.batch_size):            
            state_patch[i] = minibatch[i][0][0] # i indexes minibatch, first 0 indexes state from memory, second 0 indexes patch
            state_position[i] = minibatch[i][0][1] # i indexes minibatch, 0 indexes state from memory, 1 indexes position
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state_patch[i] = minibatch[i][3][0] # i indexes minibatch, 3 indexes next_state from memory, 0 indexes patch
            next_state_position[i] = minibatch[i][3][1] # i indexes minibatch, 3 indexes next_state from memory, 1 indexes position
            done.append(minibatch[i][4])

        state_patch = np.stack(state_patch, axis=0)
        state_position = np.stack(state_position, axis=0)
        next_state_patch = np.stack(next_state_patch, axis=0)
        next_state_position = np.stack(next_state_position, axis=0)

        # check to see if state.shape is correct
        assert state_patch.shape == (self.batch_size, self.env.patch_size, self.env.patch_size, 1), f"error, state_patch has incorrect shape. should be of shape (batch_size, patch_size, patch_size, channels). got back state_patch.shape = {state_patch.shape}"
        assert state_position.shape == (self.batch_size, 2), f"error, state_position has incorrect shape. should be of shape (batch_size, 2). got back state_position.shape = {state_position.shape}"
        assert next_state_patch.shape == (self.batch_size, self.env.patch_size, self.env.patch_size, 1), f"error, next_state_patch has incorrect shape. should be of shape (batch_size, patch_size, patch_size, channels). got back next_state_patch.shape = {next_state_patch.shape}"
        assert next_state_position.shape == (self.batch_size, 2), f"error, next_state_position has incorrect shape. should be of shape (batch_size, 2). got back next_state_position.shape = {next_state_position.shape}"

        # compute value function of current(call it target) and value function of next state(call it target_next)
        if self.positional_encoding:
            target = self.model.predict([state_patch, state_position], verbose=0)
            target_next = self.model.predict([next_state_patch, next_state_position], verbose=0)
        else:
            target = self.model.predict(state_patch, verbose=0)
            target_next = self.model.predict(next_state_patch, verbose=0)

        for i in range(self.batch_size):
            # correction on the Q value for the action used,
            # if done[i] is true, then the target should be just the final reward
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # else, use Bellman Equation
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # target = max_a' (r + gamma*Q_target_next(s', a'))
                target[i][action[i]] = reward[i] + self.gamma*np.amax(target_next[i])

        # Train the Neural Network with batches where target is the value function
        if self.positional_encoding:
            self.model.fit([state_patch, state_position], target, batch_size=self.batch_size, verbose=0)
        else:
            self.model.fit(state_patch, target, batch_size=self.batch_size, verbose=0)
    

    def burn_in_memory(self):
        print("Burning in memory...")
        for scan_id in self.burn_in:
            self.env.reset(grid_id=[scan_id, None])
            for i in range(self.env.grid_size[0]):
                for j in range(self.env.grid_size[1]):
                    self.env.current_patch = np.expand_dims(self.env.patches[i][j], axis=-1)
                    self.env.current_pos = np.array([i, j])
                    self.env.state = [self.env.current_patch, self.env.current_pos]
                    state = self.env.state

                    for action in range(self.env.action_space.n):
                        if self.env.action_mask and action != 0 and self.env.current_pos[action - 1] == self.env.grid_size[action - 1] - 1:    
                            continue
                        else:
                            next_state, reward, _, _ = self.env.step(action)
                            self.memory.append((state, action, reward, next_state, False))
                        
        print("Finished burning in memory.")


    def training(self, evaluate_val_every=10):
        if self.burn_in:
            self.burn_in_memory()
        scores = []
        found_lesion = []
        overlap_lesion = []
        overlap_lesion_err = []
        mean_val_scores = []
        val_max_acc = []
        val_overlap_acc = []
        val_overlap_rmse = []
        for e in range(self.EPISODES):
            state = self.env.reset(grid_id=[random.choice(self.train), None])
            print(f"\nEpisode: {e+1}/{self.EPISODES}, Episode ID: {self.env.grid_id}, Lesion located at: {self.env.goal_pos}")
            done = False
            score = 0
            while not done:
                action, msg = self.act(state)
                next_state, reward, terminated, truncated = self.env.step(action)
                print(f"{msg}, Next pos: {next_state[1]}, Reward: {reward}")
                done = terminated or truncated
                score += reward
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                
                if done:  
                    dateTimeObj = datetime.now()
                    timestampStr = dateTimeObj.strftime("%H:%M:%S")
                    found = np.all(self.env.current_pos == self.env.goal_pos)
                    overlap = self.env.tumor_count[self.env.current_pos[0], self.env.current_pos[1]]
                    overlap_err = self.env.tumor_count[self.env.goal_pos[0], self.env.goal_pos[1]] - overlap
                    print("episode: {}/{}, steps taken: {}/{}, score: {}, found lesion: {}, overlap: {}, overlap_error: {}, e: {:.2}, time: {}".format(e+1, self.EPISODES, self.env.total_steps, self.env.max_steps, round(score, 2), found, round(overlap, 2), round(overlap_err, 2), self.epsilon, timestampStr))
                    scores.append(score)
                    found_lesion.append(found)
                    overlap_lesion.append(overlap)
                    overlap_lesion_err.append(overlap_err)
                    if e == 0 or e % evaluate_val_every == evaluate_val_every - 1:
                        val_scores, val_found_lesion, val_overlap_lesion, val_overlap_lesion_error, val_preds, val_labels = self.test()
                        mean_val_scores.append(np.mean(val_scores))
                        val_max_acc.append(np.sum(val_found_lesion)/len(self.val))
                        val_overlap_acc.append(np.sum(np.array(val_overlap_lesion) > 0)/len(self.val))
                        val_overlap_rmse.append(np.sqrt(np.mean(np.square(val_overlap_lesion_error))))
                        print("Val results:")
                        print("Mean val score: {}, Val acc (using max lesion): {}, Val acc (using overlap lesion): {}, Val overlap RMSE: {}".format(round(mean_val_scores[-1], 2), round(val_max_acc[-1], 2), round(val_overlap_acc[-1], 2), round(val_overlap_rmse[-1], 2)))
                        for (id, scr, fnd, ov, ov_err, pred, lbl) in zip(self.val, val_scores, val_found_lesion, val_overlap_lesion, val_overlap_lesion_error, val_preds, val_labels):
                            print(f"Val env id: {id}, score: {round(scr, 2)}, found lesion: {fnd}, lesion_overlap: {round(ov, 2)}, lesion_overlap_error: {round(ov_err, 2)}, prediction: {pred}, label: {lbl}")
                        print("\n")
                self.replay()
        return scores, found_lesion, overlap_lesion, overlap_lesion_err, mean_val_scores, val_max_acc, val_overlap_acc, val_overlap_rmse


    def test(self):
        scores = []
        found_lesion = []
        overlap_lesion = []
        overlap_lesion_err = []
        prediction = []
        actual = []
        for scan_id in self.val:
            state = self.env.reset(grid_id=[scan_id, None])
            patch = state[0]
            position = state[1]
            score = 0
            done = False
            while not done:
                if self.positional_encoding:
                    q_values = self.model.predict([np.expand_dims(patch, axis=0), np.expand_dims(position, axis=0)], verbose=0)[0]
                else:
                    q_values = self.model.predict(np.expand_dims(patch, axis=0), verbose=0)[0]
            
                if self.env.action_mask: q_values = self.action_mask(q_values, position)
                action = np.argmax(q_values)

                next_state, reward, terminated, truncated = self.env.step(action)
                done = terminated or truncated
                state = next_state
                patch = state[0]
                position = state[1]
                score += reward
            scores.append(score)
            found_lesion.append(np.all(self.env.current_pos == self.env.goal_pos))
            ov = self.env.tumor_count[self.env.current_pos[0], self.env.current_pos[1]]
            overlap_lesion.append(ov)
            overlap_lesion_err.append(self.env.tumor_count[self.env.goal_pos[0], self.env.goal_pos[1]] - ov)
            prediction.append(self.env.current_pos)
            actual.append(self.env.goal_pos)
        return scores, found_lesion, overlap_lesion, overlap_lesion_err, prediction, actual