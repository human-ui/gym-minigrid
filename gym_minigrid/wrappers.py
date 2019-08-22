import math
import operator
from functools import reduce

import numpy as np
import gym
from gym import error, spaces, utils
from .minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
from .minigrid import CELL_PIXELS

class ReseedWrapper(gym.core.Wrapper):
    """
    Wrapper to always regenerate an environment with the same set of seeds.
    This can be used to force an environment to always keep the same
    configuration when reset.
    """

    def __init__(self, env, seeds=[0], seed_idx=0):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        super().__init__(env)

    def reset(self, **kwargs):
        seed = self.seeds[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        self.env.seed(seed)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class ActionBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class StateBonus(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (tuple(env.agent_pos))

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class FixedGoalReward(gym.core.Wrapper):
    """
    Always reward 1 for reaching goal, and small negative 
    reward for intermediate steps
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward > 0:
            if not done:
                raise NotImplementedError('Expecting non-zero reward only if done')
            reward = 0.
        else:
            # Standard setup - receive -1 as long as not in finish (torture:()
            reward = -1
        return obs, reward, done, info

class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']

class ImgObsOneHotWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    Use one-hot encoding, rather than numeric values.
    There are 21 channels:
    - 11 for object (0..10)
    - 7 for color (0..6)
    - 3 for state (0..2)
    """

    def __init__(self, env):
        super().__init__(env)

        self._objectLen = max(OBJECT_TO_IDX.values()) + 1
        self._colorLen = max(COLOR_TO_IDX.values()) + 1
        self._stateLen = 3
        self._channels = self._objectLen + self._colorLen + self._stateLen

        # (7, 7, 3)
        (w, h, c) = env.observation_space.spaces['image'].shape
        (self._i, self._j, _) = np.unravel_index(np.arange(w * h * c), (w, h, c))

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(w, h, self._channels), # (7, 7, 21)
            dtype='float'
        )


    def observation(self, obs):
        img = obs['image']
        img[:,:,1] += self._objectLen
        img[:,:,2] += self._objectLen + self._colorLen
        k = img.reshape((-1))

        one_hot = np.zeros(self.observation_space.shape)
        one_hot[self._i, self._j, k] = 1.
        return one_hot

class RGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width*CELL_PIXELS, self.env.height*CELL_PIXELS, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        return env.render(mode = 'rgb_array', highlight = False)

class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        return full_grid

class FullyObsOneHotWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a one-hot grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        self._maxObjectId = 9
        self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.env.width, self.env.height, self._maxObjectId + 1 + 4),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        # Just take object ID part, ignore color and state
        full_grid = self.env.grid.encode()[:,:,0]
        one_hot = np.zeros(self.observation_space.shape)
        # TODO: there must be more more efficient way to convert this
        for i in range(self.env.width):
            for j in range(self.env.height):
                one_hot[i][j][full_grid[i][j]] = 1
        # Mark agent position as (10,11,12,13) depending on direction
        agent_obj_id = self._maxObjectId + 1 + self.env.agent_dir
        one_hot[self.env.agent_pos[0]][self.env.agent_pos[1]][agent_obj_id] = 1
        return one_hot

class FlatObsWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 27

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, imgSize + self.numCharCodes * self.maxStrLen),
            dtype='uint8'
        )

        self.cachedStr = None
        self.cachedArray = None

    def observation(self, obs):
        image = obs['image']
        mission = obs['mission']

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert len(mission) <= self.maxStrLen, 'mission string too long ({} chars)'.format(len(mission))
            mission = mission.lower()

            strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype='float32')

            for idx, ch in enumerate(mission):
                if ch >= 'a' and ch <= 'z':
                    chNo = ord(ch) - ord('a')
                elif ch == ' ':
                    chNo = ord('z') - ord('a') + 1
                assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs

class AgentViewWrapper(gym.core.Wrapper):
    """
    Wrapper to customize the agent field of view size.
    """

    def __init__(self, env, agent_view_size=7):
        super(AgentViewWrapper, self).__init__(env)

        # Override default view size
        env.unwrapped.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(agent_view_size, agent_view_size, 3),
            dtype='uint8'
        )

        # Override the environment's observation space
        self.observation_space = spaces.Dict({
            'image': observation_space
        })

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
