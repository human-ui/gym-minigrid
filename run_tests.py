import sys, copy
import numpy as np
import gym

import gym_minigrid
from gym_minigrid import encoding


CH = encoding.Channels()


class Test(object):

    def __init__(self, env_name, n_envs=4, n_chars=None, seed=1337, max_steps=200):
        self.env_name = env_name
        self.n_envs = n_envs
        self.n_chars = len(self.env_name)
        self.seed = seed
        self.max_steps = max_steps
        self.n_steps = int(self.max_steps * 2.5)

    def _init_env(self, n_envs=None, seed=None):
        n_envs = self.n_envs if n_envs is None else n_envs
        seed = self.seed if seed is None else seed

        return gym.make(self.env_name, n_envs=n_envs, seed=self.seed, max_steps=self.max_steps)

    def _run_n_steps(self, env, n_steps):
        rng = np.random.RandomState(0)
        for step in range(n_steps):
            action = rng.randint(env.action_space.n, size=env.n_envs)
            obs, reward, done, info = env.step(action)
        return obs, reward, done, info

    def test_a_few_episodes(self):
        env = self._init_env()
        rng = np.random.RandomState(0)

        for step in range(self.n_steps):
            # Pick a random action
            action = rng.randint(env.action_space.n, size=env.n_envs)
            obs, reward, done, info = env.step(action)

            # Validate agent position
            assert np.all(env.agent_pos[0] < env.height)
            assert np.all(env.agent_pos[1] < env.width)

            # Check that the reward is within the specified range
            assert np.all(reward >= env.reward_range[0])
            assert np.all(reward <= env.reward_range[1])

            # verify that channels are correct
            grid = env.grid.asarray()

            # each cell is either empty or has a single object
            empty = grid[:, CH.empty]

            obj = grid[:, CH.object_type].sum(axis=1)
            assert np.all(obj[empty] == 0)
            assert np.all(obj[~empty] == 1)

            color = grid[:, CH.object_color].sum(axis=1)
            assert np.all(color[empty] == 0)
            assert np.all(color[~empty] == 1)

            # door state only set where a door is
            door = grid[:, CH.door]
            door_state = grid[:, CH.door_state].sum(axis=1)
            assert np.all(door_state[~door] == 0)
            assert np.all(door_state[door] == 1)

            # agent ocupies only a single cell
            agent_pos = grid[:, CH.agent_pos]
            assert np.all(agent_pos.sum(axis=2).sum(axis=1) == 1)

            agent_pos_idx = np.concatenate([np.nonzero(p) for p in agent_pos], axis=1).T
            assert np.all(agent_pos_idx == env.agent_pos)

            # agent state only set where the agent is
            agent_state = grid[:, CH.agent_state].sum(axis=1)
            assert np.all(agent_state[~agent_pos] == 0)
            assert np.all(agent_state[agent_pos] == 1)

            # only the agent and boxes can carry an object
            can_carry = np.logical_or(agent_pos, grid[:, CH.box])
            assert not np.any(grid[:, CH.carrying][~can_carry])
            carrying = grid[:, CH.carrying]

            carrying_obj = grid[:, CH.carrying_type].sum(axis=1)
            assert np.all(carrying_obj[carrying] == 1)
            assert np.all(carrying_obj[~carrying] == 0)
            assert np.all(carrying_obj[~can_carry] == 0)

            carrying_color = grid[:, CH.carrying_color].sum(axis=1)
            assert np.all(carrying_color[carrying] == 1)
            assert np.all(carrying_color[~carrying] == 0)
            assert np.all(carrying_color[~can_carry] == 0)

    def test_endpoint(self, n_iters=5):
        """
        Check it always ends up in the same place
        """
        for i in range(5):
            env = self._init_env()
            obs, reward, done, info = self._run_n_steps(env, self.n_steps)
            if i == 0:
                ref_obs = obs
            else:
                assert np.all(ref_obs == obs)

    def test_n_envs(self):
        """
        Check if running a single a single env equals running multiple envs
        """

        env_single = self._init_env(n_envs=1)
        env_multi = self._init_env(n_envs=4)
        env_multi.grid._grid[0] = env_single.grid._grid[0]
        env_multi.agent_pos[0] = env_single.agent_pos[0]

        # prepare a fixed sequence of actions
        rng = np.random.RandomState(0)
        actions = rng.randint(env_single.action_space.n, size=(self.n_steps, 4))

        for action in actions:
            # resets need to be controlled because randomness differs between
            # the two envs
            reset = (env_single.is_done
                     or env_multi.is_done[0]
                     or (env_single.step_count + 1 >= env_single.max_steps)
                     or (env_multi.step_count[0] + 1 >= env_multi.max_steps)
                     )
            if reset:
                env_single.reset()
                env_multi.reset()
                env_multi.grid._grid[0] = env_single.grid._grid[0]
                env_multi.agent_pos[0] = env_single.agent_pos[0]

            obs_single, _, _, _ = env_single.step(action[:1])
            obs_multi, reward, done, info = env_multi.step(action)

            assert np.all(env_single.grid[0] == env_multi.grid[0])
            assert np.all(obs_single[0] == obs_multi[0])

    def test_seed(self):
        """
        Verify that the same seed always produces the same environment
        """
        for i in range(0, 5):
            seed = 1337 + i
            env1 = self._init_env(seed=seed)
            env2 = self._init_env(seed=seed)
            assert env1 == env2

    def test_render(self):
        env = self._init_env()
        for mode in env.metadata['render.modes']:
            if mode != 'human':
                env.render(mode)

    def test_close(self):
        env = self._init_env()
        env.close()

    def test_all(self):
        print(f'Testing {self.env_name}:'.ljust(n_chars, ' '), end='', flush=True)
        for test_name in dir(self):
            test = getattr(self, test_name)
            if test_name.startswith('test_') and test_name != 'test_all' and callable(test):
                test()
                print('Y', end='', flush=True)

        print()


env_list = [e for e in gym.envs.registry.env_specs if e.startswith('MiniGrid')]
print(f'{len(env_list)} environments registered')
print()

n_chars = len(max(env_list, key=lambda x: len(x))) + 15

for env_idx, env_name in enumerate(env_list):
    Test(env_name, n_envs=4, n_chars=n_chars).test_all()
