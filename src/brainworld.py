import random
import numpy as np
import copy
from gym import Env, spaces
from utils import get_overlap, read_scan, find_tumorous_slice, patchify_slice

def find_most_tumorous_patch(patchified_seg):
    n = len(patchified_seg)
    m = len(patchified_seg[0])
    tumour_count = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            tumour_count[i, j] = np.sum(patchified_seg[i][j] != 0) # just count the number of non-zero pixels
    
    return np.array(np.unravel_index(np.argmax(tumour_count), tumour_count.shape))

def find_all_tumorous_patches(patchified_seg):
    n = len(patchified_seg)
    m = len(patchified_seg[0])
    tumour_count = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            tumour_count[i, j] = np.sum(patchified_seg[i][j] != 0) # just count the number of non-zero pixels
    
    return np.argwhere(tumour_count != 0)

class BrainWorldEnv(Env):
    def __init__(self, grid_size=(4, 4), start_pos=np.array([0, 0]), grid_id=[None, None], modality='t1ce', max_steps=20):
        assert type(start_pos) == np.ndarray, "start_pos must be a numpy array"

        super(BrainWorldEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(grid_size[0] * grid_size[1])
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.grid_id = grid_id
        self.modality = modality
        self.max_steps = max_steps
        self.total_steps = 0
        
        if grid_id[0] is None:
            self.grid_id[0] = random.choice(get_overlap())
        else:
            assert grid_id[0] in get_overlap(), "grid_id[0] must be a valid BraTS21ID Task 2 ID (in get_overlap())"
            self.grid_id[0] = grid_id[0]
        
        self.scan = read_scan(self.grid_id[0], modality=self.modality)
        self.seg = read_scan(self.grid_id[0], modality='seg')
        self.patch_size = self.scan.shape[0]//self.grid_size[0]

        if grid_id[1] is None:
            self.grid_id[1] = find_tumorous_slice(self.seg) # need to update to specify axis?
        else:
            assert grid_id[1] in range(self.scan.shape[-1]), "grid_id[1] must be a valid slice number"
            self.grid_id[1] = grid_id[1]
        
        self.patches = patchify_slice(self.scan[:, :, self.grid_id[1]], patch_size=self.patch_size, stride=self.patch_size) # need to update to specify axis?
        self.patches_seg = patchify_slice(self.seg[:, :, self.grid_id[1]], patch_size=self.patch_size, stride=self.patch_size) # need to update to specify axis?
        
        self.current_pos = copy.deepcopy(self.start_pos)
        self.current_patch = np.expand_dims(self.patches[start_pos[0]][start_pos[1]], axis=-1)
        self.state = [self.current_patch, self.current_pos]

        self.goal_pos = find_most_tumorous_patch(self.patches_seg)

    def step(self, action):

        self.total_steps += 1
        msg = f"\tSTEP: {self.total_steps}/20, Current pos: {self.current_pos}, Action taken: {action}, Goal_pos: {self.goal_pos}"

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        done = False

        if action == 0: # stay still

            if np.all(self.current_pos == self.goal_pos): # overlap lesion
                # done = True
                reward = 1
            else: # outside lesion
                reward = -2

        elif action == 1: # move down

            # assert self.current_pos[0] < self.grid_size[0] - 1, "Cannot move down from bottom row" # from old action masking
            new_pos = copy.deepcopy(self.current_pos) + np.array([1, 0])
            msg += f", new_pos: {new_pos}"
            if new_pos[0] == self.grid_size[0]:
                # wrap grid back to top
                new_pos[0] = 0
            
            if np.all(self.current_pos == self.goal_pos): # we were overlapping lesion, but we moved away
                reward = -0.5
            elif np.all(new_pos == self.goal_pos): # we are moving into lesion
                reward = 1
            else: # we weren't in lesion, and we're moving still not in lesion
                reward = -0.5

            self.current_pos[0] = new_pos[0]

        elif action == 2: # move right

            # assert self.current_pos[1] < self.grid_size[1] - 1, "Cannot move right from rightmost column" # from old action masking
            new_pos = copy.deepcopy(self.current_pos) + np.array([0, 1])
            msg += f", new_pos: {new_pos}"
            if new_pos[1] == self.grid_size[1]:
                # wrap grid back to left
                new_pos[1] = 0

            if np.all(self.current_pos == self.goal_pos): # we were overlapping lesion, but we moved away
                reward = -0.5
            elif np.all(new_pos == self.goal_pos): # we are moving into lesion
                reward = 1
            else: # we weren't in lesion, and we're moving still not in lesion
                reward = -0.5
            
            self.current_pos[1] = new_pos[1]

        else: # invalid action
            raise ValueError("Invalid action. Must be one of [0, 1, 2], where 0 = stay still, 1 = move down, 2 = move right.")

        self.current_patch = np.expand_dims(self.patches[self.current_pos[0]][self.current_pos[1]], axis=-1)
        self.state = [self.current_patch, copy.deepcopy(self.current_pos)]
        # print(msg)
        return self.state, reward, done, self.total_steps >= self.max_steps

    def reset(self, seed = None, grid_id=[None, None], modality='t1ce'):
        super().reset(seed=seed)
        if grid_id[0] is None:
            self.grid_id[0] = random.choice(get_overlap())
        else:
            assert grid_id[0] in get_overlap(), "grid_id[0] must be a valid BraTS21ID Task 2 ID (in get_overlap())"
            self.grid_id[0] = grid_id[0]
        
        self.scan = read_scan(self.grid_id[0], modality=self.modality)
        self.seg = read_scan(self.grid_id[0], modality='seg')
        self.patch_size = self.scan.shape[0]//self.grid_size[0]

        if grid_id[1] is None:
            self.grid_id[1] = find_tumorous_slice(self.seg) # need to update to specify axis?
        else:
            assert grid_id[1] in range(self.scan.shape[-1]), "grid_id[1] must be a valid slice number"
            self.grid_id[1] = grid_id[1]
        
        self.patches = patchify_slice(self.scan[:, :, self.grid_id[1]], patch_size=self.patch_size, stride=self.patch_size) # need to update to specify axis?
        self.patches_seg = patchify_slice(self.seg[:, :, self.grid_id[1]], patch_size=self.patch_size, stride=self.patch_size) # need to update to specify axis?
        
        self.current_pos = copy.deepcopy(self.start_pos)
        self.current_patch = np.expand_dims(self.patches[self.start_pos[0]][self.start_pos[1]], axis=-1)
        self.state = [self.current_patch, self.current_pos]

        self.goal_pos = find_most_tumorous_patch(self.patches_seg)
        self.total_steps = 0

        return self.state