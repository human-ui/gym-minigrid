import sys
import numpy as np
import gym
import gym_minigrid
import tqdm

rng = np.random.RandomState(0)
env = gym.make('MiniGrid-BlockedUnlockPickup-v1', seed=0)

for _ in tqdm.trange(10000, desc='obs grid'):
    obs = env[env.agent.view_box]

for _ in tqdm.trange(10000, desc='mask'):
    mask = obs.visible()

for _ in tqdm.trange(10000, desc='encode'):
    im = obs.encode(mask=mask)
    im = im[env._encoder.obs_inds]

for _ in tqdm.trange(10000, desc='rotate'):
    np.rot90(im, k=obs.agent.dir + 1, axes=(1,2))

for _ in tqdm.trange(10000, desc='step'):
    action = rng.randint(env.action_space.n)
    env.step(action)
