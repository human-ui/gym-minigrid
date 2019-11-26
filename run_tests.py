import numpy as np
import gym

from gym_minigrid import wrappers

rng = np.random.RandomState(1337)

env_list = [e for e in gym.envs.registry.env_specs if e.startswith('MiniGrid')]
print(f'{len(env_list)} environments registered')

for env_name in env_list:
    print(f'testing {env_name}')

    # Load the gym environment
    env = gym.make(env_name, seed=1337)
    print(env)

    env.max_steps = min(env.max_steps, 200)
    env.reset()
    env.render('rgb_array')

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
        action = rng.randint(0, env.action_space.n)

        obs, reward, done, info = env.step(action)

        # Validate the agent position
        assert env.agent.pos[0] < env.height
        assert env.agent.pos[1] < env.width

        # Test observation encode/decode roundtrip
        img = obs['image']
        img2 = env.decode(img).encode()
        assert np.array_equal(img, img2)

        # Check that the reward is within the specified range
        assert reward >= env.reward_range[0], reward
        assert reward <= env.reward_range[1], reward

        if done:
            num_episodes += 1
            env.reset()

        env.render('rgb_array')

    # Test the close method
    env.close()

    env = gym.make(env_name, seed=1337)
    env = wrappers.ReseedWrapper(env)
    for _ in range(10):
        env.reset()
        env.step(0)
        env.close()

    env = gym.make(env_name, seed=1337)
    env = wrappers.ImgObsWrapper(env)
    env.reset()
    env.step(0)
    env.close()

    # Test the fully observable wrapper
    env = gym.make(env_name, seed=1337)
    env = wrappers.FullyObsWrapper(env)
    env.reset()
    obs, _, _, _ = env.step(0)
    assert obs['image'].shape == env.observation_space.spaces['image'].shape
    env.close()

    env = gym.make(env_name, seed=1337)
    env = wrappers.FlatObsWrapper(env)
    env.reset()
    env.step(0)
    env.close()

    env = gym.make(env_name, seed=1337)
    env = wrappers.ViewSizeWrapper(env, 5)
    env.reset()
    env.step(0)
    env.close()

    env = gym.make(env_name, seed=1337)
    env = wrappers.ImgObsOneHotWrapper(env)
    env.reset()
    env.step(0)
    env.close()

    # Test the wrappers return proper observation spaces.
    wrapper_list = [
        wrappers.RGBImgObsWrapper,
        wrappers.RGBImgPartialObsWrapper,
    ]
    for wrapper in wrapper_list:
        env = wrapper(gym.make(env_name, seed=1337))
        obs_space, wrapper_name = env.observation_space, wrapper.__name__
        assert isinstance(
            obs_space, gym.spaces.Dict
        ), f'Observation space for {wrapper_name} is not a Dict: {obs_space}.'

        # this should not fail either
        wrappers.ImgObsWrapper(env)


print('testing the in method')
env = gym.make('MiniGrid-DoorKey-v1', seed=1337)
goal_pos = (env.grid.shape[0] - 2, env.grid.shape[1] - 2)

assert ('green', 'goal') in env.grid
assert ('blue', 'key') not in env.grid
