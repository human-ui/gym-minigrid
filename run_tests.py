import numpy as np
import gym

import gym_minigrid
# from gym_minigrid import wrappers
rng = np.random.RandomState(1337)

env_list = [e for e in gym.envs.registry.env_specs if e.startswith('MiniGrid')]
print(f'{len(env_list)} environments registered')

# env = gym.make('MiniGrid-FourRooms-v2', seed=0, see_through_walls=True)
# print(env)
# breakpoint()
# print(env)
for env_idx, env_name in enumerate(env_list):
    print(f'testing {env_name} {env_idx + 1}/{len(env_list)}')

    # Load the gym environment
    env_name = 'MiniGrid-FourRooms-v2'
    env = gym.make(env_name, seed=1337)
    print(env)

    env.max_steps = min(env.max_steps, 200)
    env.reset()
    # env.render('rgb_array')

    # Verify that the same seed always produces the same environment
    for i in range(0, 5):
        seed = 1337 + i
        env.seed(seed)
        grid1 = env.grid
        env.seed(seed)
        grid2 = env.grid
        assert grid1 == grid2

    env.reset()

    # Run for a few episodes
    num_episodes = 0
    while num_episodes < 5:
        # Pick a random action
        # action = rng.randint(env.action_space.n)
        action = rng.randint(3, size=16)

        obs, reward, done, info = env.step(action)

        # Validate the agent position
        assert np.all(env.agent_pos[0] < env.height)
        assert np.all(env.agent_pos[1] < env.width)

        # Test observation encode/decode roundtrip
        # grid = env.decode_obs(obs)
        # obs2 = grid.encode_obs()
        # assert np.array_equal(obs, obs2)

        # enc = env.encode()
        # env2 = env.decode(enc)
        # enc2 = env2.encode()
        # assert np.array_equal(enc, enc2)
        # assert env == env2

        # Check that the reward is within the specified range
        assert reward >= env.reward_range[0], reward
        assert reward <= env.reward_range[1], reward

        if done:
            num_episodes += 1
            env.reset()

        # env.render('rgb_array')

    # Test the close method
    env.close()
    breakpoint()

    # env = gym.make(env_name, seed=1337)
    # env = wrappers.ReseedWrapper(env)
    # for _ in range(10):
    #     env.reset()
    #     env.step(0)
    #     env.close()

    # env = gym.make(env_name, seed=1337)
    # env = wrappers.ImgObsWrapper(env)
    # env.reset()
    # env.step(0)
    # env.close()

    # # Test the fully observable wrapper
    # env = gym.make(env_name, seed=1337)
    # env = wrappers.FullyObsWrapper(env)
    # env.reset()
    # obs, _, _, _ = env.step(0)
    # assert obs['image'].shape == env.observation_space.spaces['image'].shape
    # env.close()

    # env = gym.make(env_name, seed=1337)
    # env = wrappers.FlatObsWrapper(env)
    # env.reset()
    # env.step(0)
    # env.close()

    # env = gym.make(env_name, seed=1337)
    # env = wrappers.ViewSizeWrapper(env, 5)
    # env.reset()
    # env.step(0)
    # env.close()

    # env = gym.make(env_name, seed=1337)
    # env = wrappers.ImgObsOneHotWrapper(env)
    # env.reset()
    # env.step(0)
    # env.close()

    # # Test the wrappers return proper observation spaces.
    # wrapper_list = [
    #     wrappers.RGBImgObsWrapper,
    #     wrappers.RGBImgPartialObsWrapper,
    # ]
    # for wrapper in wrapper_list:
    #     env = wrapper(gym.make(env_name, seed=1337))
    #     obs_space, wrapper_name = env.observation_space, wrapper.__name__
    #     assert isinstance(
    #         obs_space, gym.spaces.Dict
    #     ), f'Observation space for {wrapper_name} is not a Dict: {obs_space}.'

    #     # this should not fail either
    #     wrappers.ImgObsWrapper(env)


print('testing the in method')
env = gym.make('MiniGrid-DoorKey-v1', seed=1337)
goal_pos = (env.grid.shape[0] - 2, env.grid.shape[1] - 2)

assert ('green', 'goal') in env.grid
assert ('blue', 'key') not in env.grid
