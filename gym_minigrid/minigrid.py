import math, copy, contextlib

import numpy as np
import gym

from gym_minigrid import entities, render, encoding, window


CH = encoding.Channels()
ACTIONS = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
ROTATIONS = (('up', 'right'), ('right', 'down'), ('down', 'left'), ('left', 'up'))


class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'ansi', 'rgb_array', 'ax'],
        'video.frames_per_second': 10
    }

    class Grid(object):
        """
        Represent a grid and operations on it

        It is defined as an array of Cells.
        """

        def __init__(self, height, width, n_envs=1, view_size=7):
            self.height = height
            self.width = width
            self.padding = view_size - 1
            self._grid = np.zeros((n_envs,
                                   len(CH),
                                   height + 2 * self.padding,
                                   width + 2 * self.padding),
                                  dtype=bool)
            self._ib = np.arange(n_envs, dtype=np.intp).reshape((-1, 1, 1, 1))
            self._ich = np.arange(len(CH), dtype=np.intp).reshape((1, -1, 1, 1))
            self._idx = None  # all envs
            self.set_empty()

        def set_empty(self):
            if self._idx is None:
                envs = self._ib
            else:
                envs = self._idx

            p2 = np.arange(self.padding, self.height + self.padding).reshape(1,1,-1,1)
            p3 = np.arange(self.padding, self.width + self.padding).reshape(1,1,1,-1)
            self._grid[envs, self._ich, p2, p3] = False
            self._grid[envs, CH.empty, p2, p3] = True

        @contextlib.contextmanager
        def select(self, idx):
            """
            Masks self._grid to use only envs that match idx.

            idx must be an int or an array of indices
            """
            current_idx = self._idx
            self._idx = idx
            try:
                yield
            finally:
                self._idx = current_idx

        def __getattr__(self, attr):
            if attr in self.__dict__:
                return self.__dict__[attr]
            elif f'_{attr}' in self.__dict__:
                return self.__dict__[f'_{attr}']
            elif hasattr(self._grid, attr):
                return getattr(self.asarray(), attr)
            else:
                raise AttributeError(f'No attribute {attr}')

        def _get_pos(self, idx):
            if isinstance(idx, int):
                if self._idx is None:
                    return idx
                else:
                    return self._idx

            # if envs are not masked
            if self._idx is None:
                pos0 = idx[0]
            else:
                pos0 = self._idx

            if len(idx) >= 3:
                p2 = idx[2] + self.padding
                if len(idx) == 3:
                    pos = (pos0, idx[1], p2)
                else:
                    p3 = idx[3] + self.padding
                    pos = (pos0, idx[1], p2, p3)
                return pos
            elif len(idx) == 1:
                return (pos0,)
            elif len(idx) == 2:
                return (pos0, idx[1])

        def __getitem__(self, idx):
            return self._grid[self._get_pos(idx)]

        def __setitem__(self, idx, value):
            self._grid[self._get_pos(idx)] = value

        def asarray(self):
            if self._idx is None:
                pos0 = np.s_[:]
            else:
                pos0 = self._idx
            return self._grid[pos0, :,
                              self.padding: self.padding + self.height,
                              self.padding: self.padding + self.width]

    def __init__(
        self,
        height,
        width,
        n_envs=16,
        max_steps=100,
        seed=0,
        agent_view_size=7
    ):
        # Actions are discrete integer values
        self.action_space = gym.spaces.Discrete(len(ACTIONS))

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(n_envs, len(CH), agent_view_size, agent_view_size),
            dtype=int
        )

        # Range of possible rewards
        self._step_reward = -1
        self._win_reward = 100
        self._lose_reward = -200
        self.reward_range = (self._lose_reward, self._win_reward)

        # Environment configuration
        self.height = height
        self.width = width
        self.n_envs = n_envs
        # if n_envs == 1:
        #     raise ValueError('MiniGrid only works with more than a single environment')
        self._ib = np.arange(self.n_envs, dtype=np.intp).reshape((-1, 1, 1, 1))  # indices over batch (env) dimension
        self._ich = np.arange(len(CH), dtype=np.intp).reshape((1, -1, 1, 1))
        self.view_size = agent_view_size
        self._ivs = np.arange(self.view_size)
        self.max_steps = max_steps

        # Initialize the RNG
        self.initial_seed = seed
        self.seed(seed=seed)

        # Initialize the grid
        self.grid = self.Grid(self.height, self.width, n_envs=self.n_envs, view_size=self.view_size)
        self.agent_pos = -np.ones((self.n_envs, 2), dtype=np.intp)

        # Initialize the state
        self.step_count = np.zeros(self.n_envs, dtype=int)
        self.reward = np.zeros(self.n_envs, dtype=int)
        self.is_done = np.ones(self.n_envs, dtype=bool)
        self.reset_mask = np.zeros(self.n_envs, dtype=bool)
        # nan so that we wouldn't log cumm_reward=0
        self.cumm_reward = np.ones(self.n_envs) * np.nan

        self.reset()
        # Return first observation
        self.initial_obs = self.get_obs()

    def __str__(self):
        return self.render(mode='ansi')

    def __eq__(self, other):
        return np.array_equal(self.grid, other.grid)

    def _gen_grid(self):
        raise NotImplementedError('_gen_grid needs to be implemented by each environment')

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def reset(self, envs=None):
        # Reset step count and cummulative reward since episode start
        if envs is None:
            envs = np.ones(self.n_envs, dtype=bool)
        envs_to_reset = np.arange(self.n_envs)[envs]

        # pretend we only have a single environment for now
        agent_pos = self.agent_pos
        del self.agent_pos

        n_envs = self.n_envs
        self.n_envs = 1

        for i in envs_to_reset:
            # only update a single env inside self.grid
            with self.grid.select(i):
                self.grid.set_empty()
                self._gen_grid()
            agent_pos[i] = self.agent_pos

        # restore the original number of envs
        self.n_envs = n_envs
        self.agent_pos = agent_pos

        # reset state
        self.step_count[envs] = 0
        self.reward[envs] = 0
        self.is_done[envs] = False
        self.reset_mask[envs] = True
        self.cumm_reward[envs] = np.nan

    def seed(self, seed=0):
        # Seed the random number generator
        self.rng, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _get_slice(self, pos, channels=None, envs=None):
        if envs is None:
            envs = np.arange(self.n_envs)

        if isinstance(channels, str):  # attribute name
            channels = getattr(CH, channels)
        elif channels is None:
            channels = self._ich

        if not isinstance(channels, int):
            channels = channels.reshape(1, -1, 1, 1)
            if len(channels) == len(envs):
                channels = channels[envs]

        if isinstance(pos, np.ndarray):
            pos = np.reshape(pos, (self.n_envs, -1))
        # if pos is a tuple of indices, like (4,5), turn it into array
        elif isinstance(pos[0], int) and isinstance(pos[1], int):
            pos = np.reshape(pos, (1, -1))
        else:
            raise ValueError(f'Position {pos} not understood')

        if not isinstance(pos[:, 0], int):
            p0 = np.array(pos[:, 0]).reshape(-1, 1, 1, 1)
            if len(p0) == len(envs):
                p0 = p0[envs]
        else:
            p0 = pos[:, 0]

        if not isinstance(pos[:, 1], int):
            p1 = np.array(pos[:, 1]).reshape(-1, 1, 1, 1)
            if len(p1) == len(envs):
                p1 = p1[envs]
        else:
            p1 = pos[:, 1]

        sl = (self._ib[envs], channels, p0, p1)
        return sl

    def get_value(self, pos, channels=None, envs=None):
        sl = self._get_slice(pos, channels=channels)
        value = self.grid[sl]
        if envs is not None:
            if len(envs) > 0:
                value = value[np.array(envs)]
        value = value.squeeze()
        if self.n_envs == 1:
            value = np.array([value])
        return value

    def set_value(self, pos, value, channels=None, envs=None):
        sl = self._get_slice(pos, channels=channels, envs=envs)

        if not isinstance(value, bool):
            value = value[:, :, None, None]

        if envs is None:
            self.grid[sl] = value

        elif len(envs) > 0 and np.any(envs):
            if not isinstance(value, bool):
                self.grid[sl] = value[envs]
            else:
                self.grid[sl] = value

    def set_true(self, pos, channels=None, envs=None):
        self.set_value(pos, True, channels=channels, envs=envs)

    def set_false(self, pos, channels=None, envs=None):
        self.set_value(pos, False, channels=channels, envs=envs)

    def clear(self, pos):
        self.set_false(pos)
        self.set_true(pos, channels=CH.empty)

    def set_attr(self, pos, attr, value=None, envs=None):
        if value is None:
            self.set_true(pos, attr, envs=envs)
        elif isinstance(value, str):
            self.set_false(pos, channels=attr, envs=envs)  # set all channels within the attribute to False
            self.set_true(pos, channels=value, envs=envs)  # set target channel to True
        else:  # channels is an array of channel indices
            self.set_value(pos, value, attr, envs=envs)

        if attr == 'object_type':
            self.set_false(pos, channels='empty', envs=envs)

        # additionally store agent position for a quick agent properties lookup
        if attr == 'agent_pos':
            self.agent_pos = pos

    def set_obj(self, pos, type_, color='blue', state=None):
        default_colors = {
            'wall': 'grey',
            'goal': 'green',
            'lava': 'red'
        }
        if type_ in default_colors:
            color = default_colors[type_]
        self.set_attr(pos, 'object_type', type_)
        self.set_attr(pos, 'object_color', color)

        if type_ == 'door':
            if state is None:
                state = 'closed'
            self.set_attr(pos, 'door_state', state)

    def set_carrying_obj(self, pos, type_, color='blue'):
        self.set_attr(pos, 'object_type', type_)
        self.set_attr(pos, 'object_color', color)

    def _get_empty_pos(self,
                       top=(0,0),
                       size=None,
                       reject_fn=None,
                       max_tries=math.inf
                       ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """
        top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.height, self.width)

        num_tries = 0

        pos = np.zeros((self.n_envs, 2), dtype=np.intp)
        counter = np.zeros(self.n_envs)

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos_tmp = np.array([
                self.rng.randint(top[0], min(top[0] + size[0], self.height), size=self.n_envs),
                self.rng.randint(top[1], min(top[1] + size[1], self.width), size=self.n_envs)
            ]).T

            # if self.grid._idx == 1:
            #     breakpoint()
            envs = self.get_value(pos_tmp, channels='empty')
            pos[envs] = pos_tmp[envs]
            counter += envs.astype(int)

            if not np.all(counter > 0):
                continue

            # Don't place the object on top of another object
            # TODO: may want to consider can_overlap and can_contain cases
            # if not self.is_empty(pos):
            #     continue

            # Check if there is a filtering criterion
            if reject_fn is not None and reject_fn(self, pos):
                continue

            break

        return pos

    def place_obj(self,
                  type_,
                  color=None,
                  state=None,
                  top=(0,0),
                  size=None,
                  reject_fn=None,
                  max_tries=math.inf
                  ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """
        pos = self._get_empty_pos(top=top, size=size, reject_fn=reject_fn, max_tries=max_tries)
        self.set_obj(pos, type_, color=color, state=state)
        return pos

    def place_agent(self, top=(0,0), size=None, rand_dir=True,
                    max_tries=math.inf):
        """
        Set the agent's starting point at an empty position in the grid
        """
        pos = self._get_empty_pos(top=top, size=size, max_tries=max_tries)
        self.set_attr(pos, 'agent_pos')

        if rand_dir:
            state = self.rng.choice(CH.attrs['agent_state'])
            self.set_attr(pos, 'agent_state', state)

    def horz_wall(self, i, j, width=None):
        if width is None:
            width = self.width - j
        for jj in range(0, width):
            self.set_obj((i, j + jj), 'wall')

    def vert_wall(self, i, j, height=None):
        if height is None:
            height = self.height - i
        for ii in range(0, height):
            self.set_obj((i + ii, j), 'wall')

    def wall_rect(self, i, j, height, width):
        self.horz_wall(i, j, width)
        self.horz_wall(i + height - 1, j, width)
        self.vert_wall(i, j, height)
        self.vert_wall(i, j + width - 1, height)

    def is_inside(self, pos):
        r = np.logical_and(pos[:, 0] >= 0, pos[:, 0] < self.height)
        c = np.logical_and(pos[:, 1] >= 0, pos[:, 1] < self.width)
        return np.logical_and(r, c)

    def see_behind(self, pos):
        channels = self.get_value(pos)
        cannot_see_behind = np.logical_or(
            channels[:, CH.wall],
            np.logical_or(
                np.logical_and(channels[:, CH.door], channels[:, CH.locked]),
                np.logical_and(channels[:, CH.door], channels[:, CH.closed]),
            )
        )
        return np.logical_not(cannot_see_behind)

    def can_overlap(self, pos):
        channels = self.get_value(pos)
        can_overlap = np.logical_or(
            np.logical_and(channels[:, CH.open], channels[:, CH.door]),
            np.logical_or(channels[:, CH.goal], channels[:, CH.lava])
        )
        return can_overlap

    def can_pickup(self, pos):
        channels = self.get_value(pos)
        can_pickup = np.logical_or(
            np.logical_or(channels[:, CH.key], channels[:, CH.ball]),
            channels[:, CH.box]
        )
        return can_pickup

    def front_pos(self, pos):
        channels = self.get_value(pos)
        offset = np.zeros((self.n_envs, 2), dtype=np.intp)
        offset[channels[:, CH.right], 0] = 0
        offset[channels[:, CH.right], 1] = 1
        offset[channels[:, CH.down], 0] = 1
        offset[channels[:, CH.down], 1] = 0
        offset[channels[:, CH.left], 0] = 0
        offset[channels[:, CH.left], 1] = -1
        offset[channels[:, CH.up], 0] = -1
        offset[channels[:, CH.up], 1] = 0

        # pos might be self.agent_pos, so we do not want to update in place
        pos = np.stack([pos[:, 0] + offset[:, 0],
                        pos[:, 1] + offset[:, 1]]).T
        return pos

    # actions #############################################################

    def rotate_left(self, envs):
        rots = []
        for _, this_state in ROTATIONS:
            sel = np.logical_and(self.get_value(self.agent_pos, this_state),
                                 envs)
            rots.append(sel)

        for (next_state, this_state), sel in zip(ROTATIONS, rots):
            self.set_true(self.agent_pos, channels=next_state, envs=sel)
            self.set_false(self.agent_pos, channels=this_state, envs=sel)

    def rotate_right(self, envs):

        rots = []
        for this_state, _ in ROTATIONS:
            sel = np.logical_and(self.get_value(self.agent_pos, this_state),
                                 envs)
            rots.append(sel)

        for (this_state, next_state), sel in zip(ROTATIONS, rots):
            self.set_true(self.agent_pos, channels=next_state, envs=sel)
            self.set_false(self.agent_pos, channels=this_state, envs=sel)

    def move_forward(self, envs):
        front_pos = self.front_pos(self.agent_pos)

        envs = np.logical_and(
            envs,
            np.logical_and(
                self.is_inside(front_pos),
                np.logical_or(
                    self.get_value(front_pos, 'empty'),
                    self.can_overlap(front_pos)),
            )
            )

        self.set_false(self.agent_pos, 'agent_pos', envs)
        self.set_true(front_pos, 'agent_pos', envs)

        state = self.get_value(self.agent_pos, 'agent_state')
        self.set_value(front_pos, state, channels='agent_state', envs=envs)

        self.set_false(self.agent_pos, 'agent_state', envs)

        # move carrying objects
        is_carrying = np.logical_and(
            envs,
            self.get_value(self.agent_pos, 'carrying')
        )
        self.set_true(front_pos, 'carrying', envs=is_carrying)
        self.set_false(self.agent_pos, 'carrying', envs=is_carrying)

        self.set_attr(front_pos, 'carrying_type',
                      self.get_value(self.agent_pos, 'carrying_type'), envs=is_carrying)
        self.set_false(self.agent_pos, 'carrying_type', envs=is_carrying)

        self.set_attr(front_pos, 'carrying_color',
                      self.get_value(self.agent_pos, 'carrying_color'), envs=is_carrying)
        self.set_false(self.agent_pos, 'carrying_color', envs=is_carrying)

        # update agent's position
        agent_state = self.get_value(front_pos,'agent_state')[0]
        if np.sum(agent_state) != 1 and envs[0]:
            breakpoint()
        self.agent_pos[envs] = front_pos[envs]

        # give reward for reaching goal and penalty for stepping into lava
        is_goal = np.logical_and(envs, self.get_value(front_pos, channels='goal'))
        is_lava = np.logical_and(envs, self.get_value(front_pos, channels='lava'))

        reward = np.ones(self.n_envs) * self._step_reward
        reward[is_goal] = self._win_reward
        reward[is_lava] = self._lose_reward

        done = np.zeros(self.n_envs, dtype=bool)
        done[is_goal] = True
        done[is_lava] = True

        return reward, done

    def pickup(self, envs):
        front_pos = self.front_pos(self.agent_pos)
        envs = np.logical_and(
            envs,
            np.logical_and(
                self.can_pickup(front_pos),
                np.logical_not(self.get_value(front_pos, channels='carrying'))
                )
        )

        self._toggle_obj_carrying(front_pos, self.agent_pos, envs, from_carrying=False)

    def drop(self, envs):
        front_pos = self.front_pos(self.agent_pos)
        envs = np.logical_and(
            envs,
            np.logical_and(
                self.get_value(front_pos, channels='empty'),
                self.get_value(self.agent_pos, channels='carrying')
                )
        )

        self._toggle_obj_carrying(self.agent_pos, front_pos, envs, from_carrying=True)

    def toggle(self, envs):
        front_pos = self.front_pos(self.agent_pos)

        door = np.logical_and(
            envs,
            self.get_value(front_pos, channels='door')
        )

        # open closed doors and close opened doors
        is_open = np.logical_and(door, self.get_value(front_pos, channels='open'))
        is_closed = np.logical_and(door, self.get_value(front_pos, channels='closed'))
        self.set_attr(front_pos, 'door_state', value='closed', envs=is_open)
        self.set_attr(front_pos, 'door_state', value='open', envs=is_closed)

        # open closed doors if you have the matching color key
        door_color_channels = np.array([getattr(CH, v) for v in CH.attrs['carrying_color']])
        door_color = self.get_value(front_pos, channels=door_color_channels)
        carrying_color = self.get_value(self.agent_pos, channels='carrying_color')
        matching_key = np.logical_and(
                self.get_value(self.agent_pos, channels='carrying_key'),
                np.all(carrying_color == door_color, axis=1)
        )
        is_locked = np.logical_and(door, self.get_value(front_pos, channels='locked'))
        self.set_attr(front_pos, 'door_state', value='open',
                      envs=np.logical_and(is_locked, matching_key))

        # replace the box by its contents (can be empty too)
        box = np.logical_and(
            envs,
            self.get_value(front_pos, channels='box')
        )
        self._toggle_obj_carrying(front_pos, front_pos, box, from_carrying=True)

    def _toggle_obj_carrying(self, from_pos, to_pos, envs, from_carrying=True):
        # if from_carrying:
        #     self.set_false(self.agent_pos, channels='carrying', envs=envs)
        # else:
        #     self.set_true(self.agent_pos, channels='carrying', envs=envs)

        for kind in ['type', 'color']:
            to_channels = getattr(CH, f'carrying_{kind}')
            from_channels = np.array([getattr(CH, v) for v in CH.attrs[f'carrying_{kind}']])
            if from_carrying:
                from_channels, to_channels = to_channels, from_channels

            value = self.get_value(from_pos, channels=from_channels)
            self.set_false(from_pos, channels=from_channels, envs=envs)
            self.set_value(to_pos, value, channels=to_channels, envs=envs)

        if from_carrying:
            self.set_false(from_pos, channels='carrying', envs=envs)
            self.set_false(to_pos, channels='empty', envs=envs)
        else:
            self.set_true(from_pos, channels='empty', envs=envs)
            self.set_true(to_pos, channels='carrying', envs=envs)

    def step(self, action):
        self.step_count += 1

        not_reset = np.logical_and(~self.is_done, self.step_count < self.max_steps)
        self.reset(envs=~not_reset)

        self.rotate_left(np.logical_and(not_reset, action == 0))
        self.rotate_right(np.logical_and(not_reset, action == 1))
        reward, done = self.move_forward(np.logical_and(not_reset, action == 2))
        self.pickup(np.logical_and(not_reset, action == 3))
        self.drop(np.logical_and(not_reset, action == 4))
        self.toggle(np.logical_and(not_reset, action == 5))

        self.reward[not_reset] = reward[not_reset]
        self.is_done[not_reset] = done[not_reset]
        self.reset_mask[not_reset] = False
        self.cumm_reward[not_reset] = np.nan

        obs = self.get_obs()

        return (obs,
                self.reward,
                self.is_done,
                {'reset': self.reset_mask, 'cumm_reward': self.cumm_reward})

    def visible(self):
        return self.view_box()

    def view_box(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """
        vs = self.view_size

        p0 = np.zeros((self.n_envs, vs), dtype=np.intp)
        p1 = np.zeros((self.n_envs, vs), dtype=np.intp)

        i, j = self.agent_pos.T

        top_i = {
            'right': i - vs // 2,
            'down': i,
            'left': i - vs // 2,
            'up': i - vs + 1
        }
        top_j = {
            'right': j,
            'down': j - vs // 2,
            'left': j - vs + 1,
            'up': j - vs // 2
        }

        for ori in ['right', 'down', 'left', 'up']:
            envs = self.get_value(self.agent_pos, ori)
            p0[envs] = (self._ivs + top_i[ori][:, None])[envs]
            p1[envs] = (self._ivs + top_j[ori][:, None])[envs]
        return p0, p1

    def get_obs(self, only_image=True):
        """
        Generate the agent's view (partially observable,
        low-resolution encoding)
        """
        if not hasattr(self, 'mission'):
            raise AttributeError('environments must define a textual mission string')

        # take a slice of the environment within agent's view size
        p0, p1 = self.view_box()
        im = self.grid[
            self._ib,
            self._ich[:, CH.obs_inds],
            p0.reshape(self.n_envs, 1, self.view_size, 1),
            p1.reshape(self.n_envs, 1, 1, self.view_size)
            ]

        # orient agent upright
        for k, (state, _) in enumerate(ROTATIONS):
            envs = self.get_value(self.agent_pos, channels=state)
            im[envs] = np.rot90(im[envs], k=k, axes=(2,3))

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)

        if only_image:
            return im
        else:
            return {'image': im,
                    'direction': self.agent.state,
                    'mission': self.mission
                    }

    def render(self, mode='human', env_idx=0):
        if mode == 'human':
            win = window.Window(self)
            win.show()
        elif mode == 'ansi':
            renderer = render.ANSI()
            output = renderer.render(self.grid.asarray()[env_idx])
            return str(output)
        elif mode in ['ax', 'rgb_array']:
            renderer = render.Matplotlib()
            ax = renderer.render(self.grid.asarray()[env_idx])
            # if mode == 'rgb_array':
            #     fig = ax.get_figure()
            #     fig.canvas.draw()
            #     breakpoint()
            #     rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            #     rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #     return rgb_array
            return ax

        else:
            raise ValueError(f'Render mode {mode} not recognized')
