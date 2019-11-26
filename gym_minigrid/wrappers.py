import math
import operator
from functools import reduce

import numpy as np
import gym
from gym import spaces
from gym_minigrid import entities


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
        tup = (tuple(env.agent.pos), env.agent.state, action)

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
        tup = (tuple(env.agent.pos))

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


# class ImgObsOneHotWrapper(gym.core.ObservationWrapper):
#     """
#     Use the image as the only observation output, no language/mission.
#     Use one-hot encoding, rather than numeric values.
#     There are 21 channels:
#     - 11 for object (0..10)
#     - 7 for color (0..6)
#     - 3 for state (0..2)
#     """

#     def __init__(self, env):
#         super().__init__(env)

#         self._objectLen = len(entities.OBJECTS) + 1
#         self._colorLen = len(entities.COLORS)
#         self._stateLen = 3
#         self._channels = self._objectLen + self._colorLen + self._stateLen

#         # (3, 7, 7)
#         c, h, w = env.observation_space.spaces['image'].shape
#         _, self._i, self._j = np.unravel_index(np.arange(c * h * w), (c, h, w))
#         # breakpoint()

#         self.observation_space = spaces.Box(
#             low=0,
#             high=1,
#             shape=(self._channels, h, w),  # (18, 7, 7)
#             dtype='float'
#         )

#     def observation(self, obs):
#         img = obs['image']
#         # breakpoint()
#         img[1,:,:] += self._objectLen
#         img[2,:,:] += self._objectLen + self._colorLen
#         k = img.reshape((-1))

#         one_hot = np.zeros(self.observation_space.shape)
#         one_hot[k, self._i, self._j] = 1
#         return one_hot


class ImgObsOneHotWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.
    """

    def __init__(self, env):
        super().__init__(env)

        obs_shape = env.observation_space['image'].shape

        # Number of bits per cell
        num_bits = len(entities.OBJECTS) + 1 + len(entities.COLORS) + 3

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(num_bits, obs_shape[1], obs_shape[2]),
            dtype='float'
        )

    def observation(self, obs):
        img = obs['image']
        out = np.zeros(self.observation_space.shape, dtype='float')

        for i in range(img.shape[1]):
            for j in range(img.shape[2]):
                type_ = img[0, i, j]
                color = img[1, i, j]
                state = img[2, i, j]

                out[type_, i, j] = 1
                out[len(entities.OBJECTS) + 1 + color, i, j] = 1
                out[len(entities.OBJECTS) + 1 + len(entities.COLORS) + state, i, j] = 1
        return out


class RGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.env.width * tile_size, self.env.height * tile_size),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        rgb_img = env.render(mode='rgb_array', highlight=False, cell_pixels=self.tile_size)
        return {
            'mission': obs['mission'],
            'image': rgb_img
        }


class RGBImgPartialObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as the only observation output
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space['image'].shape
        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(3, obs_shape[0] * tile_size, obs_shape[1] * tile_size),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img_partial = env[self.agent.view_box].render(
            mode='rgb_array',
            highlight=False,
            cell_pixels=self.tile_size,
        )

        return {
            'mission': obs['mission'],
            'image': rgb_img_partial
        }


class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.env.height, self.env.width),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.encode()
        # add agent
        full_grid[:, env.agent.pos[0], env.agent.pos[1]] = np.array([
            len(entities.OBJECTS) + 1,
            entities.COLORS.index('red'),
            env.agent.dir
        ])

        return {
            'mission': obs['mission'],
            'image': full_grid
        }


class FullyObsOneHotWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a one-hot grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        self._maxObjectId = len(entities.OBJECTS) + 1
        self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self._maxObjectId + 1 + 4, self.env.height, self.env.width),
            dtype='uint8'
        )

    def observation(self, obs):
        # Just take object ID part, ignore color and state
        full_grid = self.env.grid.encode()[0]
        one_hot = np.zeros(self.observation_space.shape)
        # TODO: there must be more more efficient way to convert this
        for i in range(self.env.height):
            for j in range(self.env.width):
                one_hot[i, j][full_grid[i, j]] = 1
        # Mark agent position as (10,11,12,13) depending on direction
        agent_idx = self.env.agent.dir
        agent_obj_id = self._maxObjectId + 1 + agent_idx
        one_hot[self.env.agent.pos[0], self.env.agent.pos[1]][agent_obj_id] = 1
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


class ViewSizeWrapper(gym.core.Wrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        # Override default view size
        env.unwrapped.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, agent_view_size, agent_view_size),
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
