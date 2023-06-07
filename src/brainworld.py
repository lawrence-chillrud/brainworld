import random
import numpy as np
import copy
from gym import Env, spaces
from utils import get_overlap, read_scan, find_tumorous_slice, patchify_slice, find_most_tumorous_patch

class BrainWorldEnv(Env):
    def __init__(self, grid_size=(4, 4), start_pos=np.array([0, 0]), grid_id=[None, None], modality='t1ce', max_steps=20, action_mask=False, find_max_lesion=True, scale_rewards=False):
        assert type(start_pos) == np.ndarray, "start_pos must be a numpy array"
        super(BrainWorldEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(grid_size[0] * grid_size[1])
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.grid_id = grid_id
        self.modality = modality
        self.max_steps = max_steps
        self.action_mask = action_mask
        self.find_max_lesion = find_max_lesion
        self.scale_rewards = scale_rewards
        
        if grid_id[0] is None:
            self.grid_id[0] = random.choice(get_overlap())
        else:
            assert grid_id[0] in get_overlap(), "grid_id[0] must be a valid BraTS21ID Task 2 ID (in get_overlap())"
            self.grid_id[0] = grid_id[0]
        
        self.scan = read_scan(self.grid_id[0], modality=self.modality)
        self.seg = read_scan(self.grid_id[0], modality='seg')
        self.patch_size = self.scan.shape[0]//self.grid_size[0]

        if grid_id[1] is None:
            self.grid_id[1] = find_tumorous_slice(self.seg)
        else:
            assert grid_id[1] in range(self.scan.shape[-1]), "grid_id[1] must be a valid slice number"
            self.grid_id[1] = grid_id[1]
        
        self.patches = patchify_slice(self.scan[:, :, self.grid_id[1]], patch_size=self.patch_size, stride=self.patch_size)
        self.patches_seg = patchify_slice(self.seg[:, :, self.grid_id[1]], patch_size=self.patch_size, stride=self.patch_size)
        
        self.current_pos = copy.deepcopy(self.start_pos)
        self.current_patch = np.expand_dims(self.patches[start_pos[0]][start_pos[1]], axis=-1)
        self.state = [self.current_patch, self.current_pos]

        self.goal_pos, self.tumor_count = find_most_tumorous_patch(self.patches_seg)
        self.tumor_count = self.tumor_count / np.sum(self.tumor_count) * 100
        self.total_steps = 0

    def step(self, action):
        '''
        Actions
        -------
        0 = stay still
        1 = move down
        2 = move right
        '''
        self.total_steps += 1
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        done = False

        if action == 0: # stay still
            overlapping_lesion = self.tumor_count[self.current_pos[0]][self.current_pos[1]] > 0
            if self.find_max_lesion:
                overlapping_lesion = np.all(self.current_pos == self.goal_pos)
            if overlapping_lesion:
                reward = 1
                if self.scale_rewards:
                    reward *= self.tumor_count[self.current_pos[0]][self.current_pos[1]]/100
            else:
                reward = -2
        else: # move down (action = 1) or right (action = 2)
            if self.action_mask: assert self.current_pos[action - 1] < self.grid_size[action - 1] - 1, "Cannot move off grid"
            to_add = np.array([0, 0])
            to_add[action - 1] = 1
            new_pos = copy.deepcopy(self.current_pos) + to_add
            if new_pos[action - 1] == self.grid_size[action - 1]:
                new_pos[action - 1] = 0 # wrap grid back to left/top
            
            moved_away_from_lesion = self.tumor_count[self.current_pos[0]][self.current_pos[1]] > 0 and self.tumor_count[new_pos[0]][new_pos[1]] == 0
            moved_into_lesion = self.tumor_count[new_pos[0]][new_pos[1]] > 0

            if self.find_max_lesion:
                moved_away_from_lesion = np.all(self.current_pos == self.goal_pos)
                moved_into_lesion = np.all(new_pos == self.goal_pos)

            if moved_away_from_lesion:
                reward = -0.5
                if self.scale_rewards:
                    reward -= self.tumor_count[self.current_pos[0]][self.current_pos[1]]/100
            elif moved_into_lesion: 
                reward = 1
                if self.scale_rewards:
                    reward *= self.tumor_count[new_pos[0]][new_pos[1]]/100
            else: # we weren't in lesion, and we're moving to a patch still not in lesion
                reward = -0.5

            self.current_pos[action - 1] = new_pos[action - 1]

        self.current_patch = np.expand_dims(self.patches[self.current_pos[0]][self.current_pos[1]], axis=-1)
        self.state = [self.current_patch, copy.deepcopy(self.current_pos)]
        
        return self.state, reward, done, self.total_steps >= self.max_steps

    def reset(self, seed = None, grid_id=[None, None]):
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
            self.grid_id[1] = find_tumorous_slice(self.seg)
        else:
            assert grid_id[1] in range(self.scan.shape[-1]), "grid_id[1] must be a valid slice number"
            self.grid_id[1] = grid_id[1]
        
        self.patches = patchify_slice(self.scan[:, :, self.grid_id[1]], patch_size=self.patch_size, stride=self.patch_size)
        self.patches_seg = patchify_slice(self.seg[:, :, self.grid_id[1]], patch_size=self.patch_size, stride=self.patch_size)
        
        self.current_pos = copy.deepcopy(self.start_pos)
        self.current_patch = np.expand_dims(self.patches[self.start_pos[0]][self.start_pos[1]], axis=-1)
        self.state = [self.current_patch, self.current_pos]

        self.goal_pos, self.tumor_count = find_most_tumorous_patch(self.patches_seg)
        self.tumor_count = self.tumor_count / np.sum(self.tumor_count) * 100
        self.total_steps = 0

        return self.state