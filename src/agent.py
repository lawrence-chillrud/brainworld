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
        self.env = BrainWorldEnv(grid_size=(4, 4), start_pos=[0, 0], grid_id=[None, None], modality='t1ce')
        
        self.action_size = self.env.action_space.n
        self.EPISODES = 90
        self.memory = deque(maxlen=2000) # 15,000?
        
        self.gamma = 0.99    # discount rate
        self.epsilon = 0.7  # exploration rate
        self.epsilon_min = 1e-4
        self.epsilon_decay = 1 - 1e-4 # decay rate on right
        self.batch_size = 128
        self.train_start = 1000

        # create main model
        self.model = VanillaCNN(input_shape=self.env.state.shape, action_space = self.action_size)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    
    def act(self, state):
        # implement the epsilon-greedy policy
        if random.random() < self.epsilon:
            # explore
            # TODO: implement action masking to ensure we don't take an illegal action!
            return self.env.action_space.sample()
        else:
            # exploit
            # TODO: implement action masking to ensure we don't take an illegal action!
            return np.argmax(self.model.predict([np.expand_dims(state, axis=0), self.env.current_pos])[0])

    
    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.empty(self.batch_size, dtype=object)
        next_state = np.empty(self.batch_size, dtype=object)
        action, reward, done = [], [], []

        # assign data into state, next_state, action, reward and done from minibatch
        for i in range(self.batch_size):            
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # TODO: check to see if state.shape is correct. 
        # maybe use something like np.stack instead to eventually make
        # state.shape = (batch_size, 60, 60, 1)
        print("\n\nLINE 67 of AGENT.PY: STATE SHAPE", state.shape)
        print("\n\n")

        # compute value function of current(call it target) and value function of next state(call it target_next)
        # TODO: implement action masking here too
        target = self.model.predict([state, self.env.current_pos])

        # Q: should we be using self.env.current_pos + action here? or no?
        # TODO: solve above question, implement action masking here too
        target_next = self.model.predict([next_state, self.env.current_pos])

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
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)


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