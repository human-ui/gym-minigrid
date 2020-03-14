import math, copy

import numpy as np
import gym

from gym_minigrid import entities, render_text, encoding
from gym_minigrid.encoding import ATTRS


CH = encoding.Channels()
ACTIONS = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
ROTATIONS = (('up', 'right'), ('right', 'down'), ('down', 'left'), ('left', 'up'))


class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'pixmap', 'ascii', 'ansi', 'curses4bit', 'curses8bit'],
        'video.frames_per_second': 10
    }

    class Grid(object):
        """
        Represent a grid and operations on it

        It is defined as an array of Cells.
        """

        def __init__(self, height, width, n_envs=1, view_size=7):
            self.padding = view_size - 1
            self._grid = np.zeros((n_envs,
                                   len(CH),
                                   height + 2 * self.padding,
                                   width + 2 * self.padding),
                                  dtype=bool)
            envs = np.arange(n_envs).reshape(-1,1,1,1)
            p2 = np.arange(self.padding, height + self.padding).reshape(1,1,-1,1)
            p3 = np.arange(self.padding, width + self.padding).reshape(1,1,1,-1)
            self._grid[envs, CH.empty, p2, p3] = True
            self._p2 = np.arange(self._grid.shape[2])
            self._p3 = np.arange(self._grid.shape[3])

        def __getattr__(self, attr):
            if attr in self.__dict__:
                return self.__dict__[attr]
            elif f'_{attr}' in self.__dict__:
                return self.__dict__[f'_{attr}']
            elif hasattr(self._grid, attr):
                return getattr(self._grid, attr)
            else:
                raise AttributeError(f'No attribute {attr}')

        def _get_pos(self, pos):
            if isinstance(pos, (tuple, list)) and len(pos) >= 3:
                p2 = pos[2] + self.padding
                if len(pos) == 3:
                    pos = (pos[0], pos[1], p2)
                else:
                    p3 = pos[3] + self.padding
                    pos = (pos[0], pos[1], p2, p3)
            return pos

        def __getitem__(self, pos):
            return self._grid[self._get_pos(pos)]

        def __setitem__(self, pos, value):
            self._grid[self._get_pos(pos)] = value

    def __init__(
        self,
        height,
        width,
        n_envs=16,
        max_steps=100,
        see_through_walls=False,
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
        self._lose_reward = -100
        self.reward_range = (self._lose_reward, self._win_reward)

        # Environment configuration
        self.height = height
        self.width = width
        self.n_envs = n_envs
        self._ib = np.arange(self.n_envs, dtype=np.intp).reshape((-1, 1, 1, 1))  # indices over batch (env) dimension
        self._ich = np.arange(len(CH), dtype=np.intp).reshape((1, -1, 1, 1))
        self.view_size = agent_view_size
        self._ivs = np.arange(self.view_size)
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Initialize the RNG
        self.initial_seed = seed
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

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

    def reset(self):
        self.agent_pos = -np.ones((2, self.n_envs), dtype=np.intp)
        self._gen_grid()
        assert np.all(self.agent_pos != -1)

        # Step count since episode start
        # FIXME: must only reset envs that are done
        self.step_count = np.zeros(self.n_envs, dtype=int)

        # Return first observation
        obs = self.get_obs()
        return obs

    def seed(self, seed=0):
        # Seed the random number generator
        self.rng, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def visible(self):
        """
        Process occluders and visibility
        Note that this incurs some performance cost  # TODO: verify if still true
        """
        def _flood_fill(pos):
            visited.add(pos)

            if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.shape[0] or pos[1] >= self.shape[1]:
                return

            if mask[pos]:  # already visited
                return

            if view_box[pos] == 0:  # outside agent's view_size
                return

            # TODO: this may be slow
            if not self.see_behind(pos):  # this is a boundary
                mask[pos] = True  # add it to the list of visibles and exit
                return

            mask[pos] = True

            # visit all neighbors in the surrounding plus
            for i, j in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                new_pos = (pos[0] + i, pos[1] + j)
                if new_pos not in visited:
                    _flood_fill(new_pos)

        view_box = np.ones(self.shape, dtype=bool)
        # from_, to_ = self._slices(self.agent.view_box)
        # view_box[from_[0], from_[1]] = True

        if self.see_through_walls:
            mask = np.ones(self.shape, dtype=bool)
        else:
            raise NotImplementedError
            mask = np.zeros(self.shape, dtype=bool)
            visited = set()
            _flood_fill(self.agent.pos)
        return mask

    ############################################################################

    def _get_slice(self, pos, channels=None):
        if isinstance(channels, str):  # attribute name
            channels = getattr(CH, channels)
        elif isinstance(channels, (tuple, list)):
            channels = np.array([getattr(CH, c) for c in channels])
        elif channels is None:
            channels = self._ich
        else:
            channels = np.array(channels)

        if not isinstance(channels, int):
            channels = channels.reshape(1, -1, 1, 1)
        else:
            channels = np.array([channels]).reshape(1, -1, 1, 1)

        if not isinstance(pos[0], int):
            p0 = np.array(pos[0]).reshape(-1, 1, 1, 1)
        else:
            p0 = pos[0]  # np.array([pos[0]]).reshape(-1, 1, 1, 1)

        if not isinstance(pos[1], int):
            p1 = np.array(pos[1]).reshape(-1, 1, 1, 1)
        else:
            p1 = pos[1]  # np.array([pos[1]]).reshape(1, 1, 1, -1)

        sl = (self._ib, channels, p0, p1)
        return sl

    def get_value(self, pos, channels=None, envs=None):
        sl = self._get_slice(pos, channels=channels)
        value = self.grid[sl]
        if envs is not None:
            if len(envs) > 0:
                value = value[np.array(envs)]
        return value.squeeze()

    def set_value(self, pos, value, channels=None, envs=None):
        sl = self._get_slice(pos, channels=channels)

        if not isinstance(value, bool):
            value = value[:, :, None, None]

        if envs is None:
            self.grid[sl] = value
        else:
            if not isinstance(value, bool):
                self.grid[sl][envs] = value[envs]
            else:
                self.grid[sl][envs] = value

    def set_true(self, pos, channels=None, envs=None):
        self.set_value(pos, True, channels=channels, envs=envs)

    def set_false(self, pos, channels=None, envs=None):
        self.set_value(pos, False, channels=channels, envs=envs)

    def clear(self, pos):
        self.set_false(pos)
        self.set_true(pos, channels=CH.empty)

    def set_attr(self, pos, attr, value=None, envs=None):
        if value is not None:
            self.set_false(pos, attr, envs=envs)  # set all channels within the attribute to False
            self.set_true(pos, value, envs=envs)  # set target channel to True
        else:
            self.set_true(pos, attr, envs=envs)
        if attr == 'object_type':
            self.set_false(pos, 'empty', envs=envs)

        # additionally store agent position for a quick agent properties lookup
        if attr == 'agent_pos':
            self.agent_pos[envs] = pos[envs]

    def set_obj(self, pos, type_, color=None, state=None):
        default_colors = {
            'wall': 'grey',
            'goal': 'green',
            'lava': 'red'
        }
        if color is None:
            color = default_colors.get(type_, 'blue')
        self.set_attr(pos, 'object_type', type_)
        self.set_attr(pos, 'object_color', color)

        if type_ == 'door':
            if state is None:
                state = 'closed'
            self.set_attr(pos, 'object_state', state)

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

        For single environment only

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """
        top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.height, self.width)

        num_tries = 0

        pos = np.zeros((2, self.n_envs), dtype=np.intp)
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
            ])

            # breakpoint()
            envs = self.get_value(pos_tmp, channels='empty')
            pos[:, envs] = pos_tmp[:, envs]
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

        if type_ == 'agent':
            self.set_attr(pos, 'agent_pos')
        else:
            self.set_obj(pos, type_, color=color, state=state)
        return pos

    def place_agent(self, top=(0,0), size=None, rand_dir=True,
                    max_tries=math.inf):
        """
        Set the agent's starting point at an empty position in the grid
        """
        pos = self.place_obj('agent', top=top, size=size, max_tries=max_tries)
        if rand_dir:
            state = self.rng.choice(CH.agent_state, size=self.n_envs)
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
        r = np.logical_and(pos[0] >= 0, pos[0] < self.height)
        c = np.logical_and(pos[1] >= 0, pos[1] < self.width)
        return np.logical_and(r, c)

    def is_empty(self, pos):
        return np.all(self.get_value(pos) == 0, axis=1)

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
        offset = np.zeros((2, self.n_envs), dtype=np.intp)
        offset[0, channels[:, CH.right]] = 0
        offset[1, channels[:, CH.right]] = 1
        offset[0, channels[:, CH.down]] = 1
        offset[1, channels[:, CH.down]] = 0
        offset[0, channels[:, CH.left]] = 0
        offset[1, channels[:, CH.left]] = -1
        offset[0, channels[:, CH.up]] = -1
        offset[1, channels[:, CH.up]] = 0
        return (pos[0] + offset[0], pos[1] + offset[1])

    # actions #############################################################

    def rotate_left(self, action):
        envs = action == 0
        for next_state, this_state in ROTATIONS:
            sel = np.logical_and(self.get_value(self.agent_pos, this_state),
                                 envs)
            self.set_true(self.agent_pos, next_state, sel)
            self.set_false(self.agent_pos, this_state, sel)

    def rotate_right(self, action):
        envs = action == 1
        for this_state, next_state in ROTATIONS:
            sel = np.logical_and(self.get_value(self.agent_pos, this_state),
                                 envs)
            self.set_true(self.agent_pos, next_state, sel)
            self.set_false(self.agent_pos, this_state, sel)

    def move_forward(self, action):
        front_pos = self.front_pos(self.agent_pos)
        envs = np.logical_and(
            action == 2,
            np.logical_and(
                self.is_inside(front_pos),
                np.logical_or(
                    self.is_empty(front_pos),
                    self.can_overlap(front_pos)),
            )
            )
        self.set_false(self.agent_pos, 'agent_pos', envs)
        self.set_true(front_pos, 'agent_pos', envs)
        self.agent_pos[0][envs] = front_pos[0][envs]
        self.agent_pos[1][envs] = front_pos[1][envs]

        is_goal = np.logical_and(envs, self.get_value(front_pos, channels='goal'))
        is_lava = np.logical_and(envs, self.get_value(front_pos, channels='lava'))

        reward = np.ones(self.n_envs) * self._step_reward
        reward[is_goal] = self._win_reward
        reward[is_lava] = self._lose_reward

        done = np.zeros(self.n_envs, dtype=bool)
        done[is_goal] = True
        done[is_lava] = True

        return reward, done

    def pickup(self, action):
        front_pos = self.front_pos(self.agent_pos)
        envs = np.logical_and(
            action == 3,
            np.logical_and(
                self.can_pickup(front_pos),
                np.logical_not(self.get_value(front_pos, channels='carrying'))
                )
        )

        self._toggle_obj_carrying(front_pos, self.agent_pos, envs, from_carrying=False)

    def drop(self, action):
        front_pos = self.front_pos(self.agent_pos)
        envs = np.logical_and(
            action == 4,
            np.logical_and(
                self.is_empty(front_pos),
                self.get_value(self.agent_pos, channels='carrying')
                )
        )

        self._toggle_obj_carrying(self.agent_pos, front_pos, envs, from_carrying=True)

    def toggle(self, action):
        front_pos = self.front_pos(self.agent_pos)
        envs = np.logical_and(
            action == 5,
            np.logical_and(
                self.is_empty(front_pos),
                self.get_value(self.agent_pos, channels='carrying')
                )
        )

        door = np.logical_and(
            envs,
            self.get_value(front_pos, channels='door')
        )
        is_locked = np.logical_and(door, self.get_value(front_pos, channels='locked'))
        is_open = np.logical_and(door, self.get_value(front_pos, channels='open'))
        is_closed = np.logical_and(door, self.get_value(front_pos, channels='closed'))

        color_channels = CH.attrs[f'carrying_color']
        color = self.get_value(front_pos, channels=color_channels)
        carrying_color = self.get_value(front_pos, channels='carrying_color')
        key_matches = np.logical_and(
            self.get_value(front_pos, channels='carrying_key'),
            np.all(carrying_color == color, axis=1)
            )

        # unlock or, if unlocked, open / close
        self.set_attr(front_pos, 'door_state', value='open',
                      envs=np.logical_and(is_locked, key_matches))
        self.set_attr(front_pos, 'door_state', value='closed', envs=is_open)
        self.set_attr(front_pos, 'door_state', value='open', envs=is_closed)

        box = np.logical_and(
            envs,
            self.get_value(front_pos, channels='box')
        )
        # replace the box by its contents (can be empty too)
        self._toggle_obj_carrying(front_pos, front_pos, box, from_carrying=True)

    def _toggle_obj_carrying(self, from_pos, to_pos, envs, from_carrying=True):
        if from_carrying:
            self.set_false(self.agent_pos, channels='carrying', envs=envs)
        else:
            self.set_true(self.agent_pos, channels='carrying', envs=envs)

        for kind in ['type', 'color']:
            to_channels = CH.attrs[f'carrying_{kind}']
            from_channels = [f'carrying_{v}' for v in to_channels]
            if not from_carrying:
                from_channels, to_channels = to_channels, from_channels

            value = self.get_value(from_pos, channels=from_channels)
            self.set_false(from_pos, channels=from_channels, envs=envs)
            self.set_value(to_pos, value, channels=to_channels, envs=envs)

    def step(self, action):
        self.step_count += 1

        self.rotate_left(action)
        self.rotate_right(action)
        reward, done = self.move_forward(action)
        self.pickup(action)
        self.drop(action)
        self.toggle(action)

        done[self.step_count >= self.max_steps] = True

        obs = self.get_obs()

        return obs, reward, done, {}

    def view_box(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """
        vs = self.view_size

        p0 = np.zeros((self.n_envs, vs), dtype=np.intp)
        p1 = np.zeros((self.n_envs, vs), dtype=np.intp)

        i, j = self.agent_pos

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
            envs = self.get_value((i, j), ori)
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

    def render(self, mode='human', highlight=True, cell_pixels=32):
        if mode in ['human', 'rgb_array', 'pixmap']:
            from gym_minigrid.render_image import MiniGridImage
            renderer = MiniGridImage(self, highlight=highlight, cell_pixels=cell_pixels)
            if mode == 'human':
                renderer.show()
            elif mode == 'rgb_array':
                return renderer.array()
            elif mode == 'pixmap':
                return renderer.pixmap()

        elif mode == 'ascii':
            return str(render_text.ASCII(self))
        elif mode == 'ansi':
            return str(render_text.ANSI(self.grid[0]))
        elif mode == 'curses4bit':
            return render_text.Curses4bit(self)()
        elif mode == 'curses8bit':
            return render_text.Curses8bit(self)()
        else:
            raise ValueError(f'Render mode {mode} not recognized')
