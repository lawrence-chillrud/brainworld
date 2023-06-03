from datetime import datetime
import random
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import random
from models import VanillaCNN
from brainworld import BrainWorldEnv

class DQNAgent:
    def __init__(self):
        self.env = BrainWorldEnv(grid_size=(4, 4), start_pos=np.array([0, 0]), grid_id=[None, None], modality='t1ce')
        
        self.action_size = self.env.action_space.n
        self.EPISODES = 90
        self.memory = deque(maxlen=2000) # 15,000?
        
        self.gamma = 0.99    # discount rate
        self.epsilon = 0.7  # exploration rate
        self.epsilon_min = 1e-4
        self.epsilon_decay = 1 - 1e-4 # decay rate on right
        self.batch_size = 128
        self.train_start = 1000 # TODO: figure out how to burn this in before replay is used in every single episode..!

        # create main model
        self.model = VanillaCNN(input_shape=self.env.state[0].shape, action_space = self.action_size)


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
        Implements the epsilon-greedy policy, with action masking
        '''
        patch = state[0]
        position = state[1]

        # implement the epsilon-greedy policy
        if random.random() < self.epsilon:
            # explore
            return self.sample_action_space(position)
        else:
            # exploit
            q_values = self.model.predict([np.expand_dims(patch, axis=0), np.expand_dims(position, axis=0)])[0]
            q_values_masked = self.action_mask(q_values, position)
            return np.argmax(q_values_masked)

    
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
        
        # check to see if state.shape is correct. 
        # if these fail, maybe use something like np.stack instead to eventually make it work?
        assert state_patch.shape == (self.batch_size, self.env.patch_size, self.env.patch_size, 1), f"error, state_patch has incorrect shape. should be of shape (batch_size, patch_size, patch_size, channels). got back state_patch.shape = {state_patch.shape}"
        assert state_position.shape == (self.batch_size, 2), f"error, state_position has incorrect shape. should be of shape (batch_size, 2). got back state_position.shape = {state_position.shape}"
        assert next_state_patch.shape == (self.batch_size, self.env.patch_size, self.env.patch_size, 1), f"error, next_state_patch has incorrect shape. should be of shape (batch_size, patch_size, patch_size, channels). got back next_state_patch.shape = {next_state_patch.shape}"
        assert next_state_position.shape == (self.batch_size, 2), f"error, next_state_position has incorrect shape. should be of shape (batch_size, 2). got back next_state_position.shape = {next_state_position.shape}"

        # compute value function of current(call it target) and value function of next state(call it target_next)
        # TODO: implement action masking here too
        target = self.model.predict([state_patch, state_position])

        # TODO: implement action masking here too
        target_next = self.model.predict([next_state_patch, next_state_position])

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
        self.model.fit([state_patch, state_position], target, batch_size=self.batch_size, verbose=0)


    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)
            
    def training(self):
        scores = []
        found_lesion = []
        for e in range(self.EPISODES):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                # if you have graphic support, you can render() to see the animation. 
                # self.env.render()
                action = self.act(state)
                next_state, reward, terminated, truncated = self.env.step(action)
                done = terminated or truncated
                score += reward
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                
                if done:  
                    dateTimeObj = datetime.now()
                    timestampStr = dateTimeObj.strftime("%H:%M:%S")
                    print("episode: {}/{}, steps taken: {}/{}, score: {}, found lesion: {}, e: {:.2}, time: {}".format(e+1, self.EPISODES, self.env.total_steps, self.env.max_steps, score, terminated, self.epsilon, timestampStr))
                    scores.append(score)
                    found_lesion.append(terminated)
                    # save model option
                    # if i >= 500:
                    #     print("Saving trained model as cartpole-dqn-training.h5")
                    #     self.save("./save/cartpole-dqn-training.h5")
                    #     return # remark this line if you want to train the model longer
                self.replay()
        return scores, found_lesion

    # test function if you want to test the learned model
    # TODO: not updated this for new environment, model, etc.
    def test(self):
        self.load("./save/cartpole-dqn-training.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                # self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e+1, self.EPISODES, i))
                    break