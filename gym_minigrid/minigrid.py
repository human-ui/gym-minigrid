import math, copy

import numpy as np
import gym

from gym_minigrid import entities, render_text


class Cell(object):

    def __init__(self):
        self.color = 'black'  # at the moment, we restrict background color to black
        self.clear()

    def clear(self):
        self.entity = None

    def __str__(self):

        if self.entity is not None:
            r = f', has:\n  {self.entity}'
        else:
            r = ', empty'
        return f'cell: {self.color}{r}'

    def copy(self):
        return copy.deepcopy(self)

    def to_array(self):
        if self.entity is None:
            arr = np.array([None, None, None])
        else:
            arr = self.entity.to_array()
        return arr

    def to_idx_array(self):
        if self.entity is None:
            arr = np.array([0, 0, 0])
        else:
            arr = self.entity.to_idx_array()
        return arr


class Grid(object):
    """
    Represent a grid and operations on it

    It is defined as an array of Cells.
    """

    def __init__(self, height, width):
        self._grid = np.empty((height, width), dtype=np.object)
        for (i, j), _ in np.ndenumerate(self._grid):
            self._grid[i, j] = Cell()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif f'_{attr}' in self.__dict__:
            return self.__dict__[f'_{attr}']
        elif hasattr(self._grid, attr):
            return getattr(self._grid, attr)
        else:
            raise AttributeError(f'No attribute {attr}')

    def __getitem__(self, pos):
        return self._grid[tuple(pos)]

    def __setitem__(self, pos, obj):
        self._grid[tuple(pos)] = obj

    def __contains__(self, key):
        if isinstance(key, entities.WorldObj):
            for cell in self._grid.ravel():
                if cell is key:
                    return True
        elif isinstance(key, tuple):
            for cell in self._grid.ravel():
                if cell.entity is not None:
                    if (cell.entity.color, cell.entity.type) == key:
                        return True
                    if key[0] is None and key[1] == cell.entity.type:
                        return True
        return False

    def __ne__(self, other):
        return not self == other

    @property
    def shape(self):
        return self._grid.shape


class Grid2(Grid):

    def __init__(self, height, width, agent):
        super().__init__(height, width)
        self.agent = agent

    def encode(self, mask=None, as_str=False):
        """
        Produce a compact numpy encoding of the grid
        """
        if mask is None:
            mask = np.ones(self._grid.shape, dtype=bool)

        n_attrs = len(self._grid[0,0].to_array())
        assert n_attrs == 3
        array = np.zeros((n_attrs,) + self._grid.shape)
        if as_str:
            array = array.astype(np.object)
        else:
            array = array.astype('uint8')

        for (i, j), cell in np.ndenumerate(self._grid):
            if as_str:
                array[:, i, j] = cell.to_array()  # type, color, state
            else:
                array[:, i, j] = cell.to_idx_array()

        return array

    @staticmethod
    def decode(array):
        """
        Decode an observation back into a grid
        """
        channels, height, width = array.shape
        assert channels == 3
        assert height == width

        agent = entities.Agent(view_size=height)
        agent.pos = (height, width // 2)
        agent.state = 'up'

        grid = Grid(height, width, agent)

        for (i, j), _ in np.ndenumerate(grid):
            type_, color, state = array[:, i, j]
            obj = entities.make(entities.OBJECTS[type_],
                                color=entities.COLORS[color])
            if hasattr(obj, 'STATES'):
                obj.state = obj.STATES[state]
            grid[i, j].entity = obj

        return grid

    def get_obs(self, as_str=False, see_through_walls=False):
        # obs_box = self.get_obs_box()
        grid = self[self.agent.view_box]
        obs = grid.encode(as_str=as_str)

        # remove the parts of environment that the agent cannot see
        mask = grid.visible(see_through_walls=see_through_walls)
        obs[:, ~mask] = None if as_str else 0

        # obs_arr = np.zeros((len(obs), self.agent.view_size, self.agent.view_size))
        # obs_arr = self.put_obs(obs_arr, obs)

        # orient agent upright
        obs = np.rot90(obs, k=self.agent.dir + 1, axes=(1,2))
        return obs

    def visible(self, see_through_walls=False):
        """
        Process occluders and visibility
        Note that this incurs some performance cost  # TODO: verify if still true
        """
        def _flood_fill(pos):
            if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.shape[0] or pos[1] >= self.shape[1]:
                return

            if mask[pos]:  # already visited
                return

            if view_box[pos] == 0:  # outside agent's view_size
                return

            if self[pos].entity is not None:
                if not self[pos].entity.see_behind():   # this is a boundary
                    mask[pos] = True  # add it to the list of visibles and exit
                    return

            mask[pos] = True

            # visit all neighbors in the surrounding square
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i != 0 or j != 0:  # skip current pos
                        _flood_fill((pos[0] + i, pos[1] + j))

        if see_through_walls:
            mask = np.ones(self.shape, dtype=bool)
        else:
            mask = np.zeros(self.shape, dtype=bool)
            _flood_fill(self.agent.pos)

        view_box = np.zeros(self.shape, dtype=bool)
        from_, to_ = self._slices(self.agent.view_box)
        view_box[from_[0], from_[1]] = True

        # obsgrid_mask = obsgrid.mask()
        # mask[top_i: bottom_i, top_j: bottom_j] = obsgrid_mask[offset_i: offset_i + bottom_i - top_i, offset_j: offset_j + bottom_j - top_j]
        return mask

    def visited(self):
        arr = np.zeros(self.shape, dtype=bool)
        for pos in self.agent.visited:
            arr[pos] = True
        return arr

    # def put_obs(self, obs):
    #     top_i, top_j, bottom_i, bottom_j = self.agent.view_box
    #     offset_i = -top_i if top_i < 0 else 0
    #     offset_j = -top_j if top_j < 0 else 0
    #     top_i = max(0, top_i)
    #     top_j = max(0, top_j)
    #     bottom_i = min(self.shape[0], bottom_i)
    #     bottom_j = min(self.shape[1], bottom_j)

    #     obs_box = slice(top_i, bottom_i), slice(top_j, bottom_j)

    #     self[obs_box] = obs[offset_i: offset_i + bottom_i - top_i,
    #                         offset_j: offset_j + bottom_j - top_j]

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
            return render_text.ANSI(self)()
        elif mode == 'curses4bit':
            return render_text.Curses4bit(self)()
        elif mode == 'curses8bit':
            return render_text.Curses8bit(self)()
        else:
            raise ValueError(f'Render mode {mode} not recognized')


class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'pixmap', 'ascii', 'ansi', 'curses4bit', 'curses8bit'],
        'video.frames_per_second': 10
    }

    def __init__(
        self,
        height,
        width,
        max_steps=100,
        see_through_walls=False,
        seed=1337,
        agent_view_size=7
    ):
        self.agent = entities.Agent(view_size=agent_view_size)
        # Action enumeration for this environment
        self.actions = entities.Agent.ACTIONS

        # Actions are discrete integer values
        self.action_space = gym.spaces.Discrete(len(self.actions))

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, agent_view_size, agent_view_size),
            dtype='uint8'
        )
        self.observation_space = gym.spaces.Dict({
            'image': self.observation_space
        })

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Environment configuration
        self.height = height
        self.width = width
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def __str__(self):
        return self.render(mode='ansi')

    def __eq__(self, other):
        env1 = self.encode()
        env2 = other.encode()
        return np.array_equal(env1, env2)

    def _gen_grid(self, height, width):
        raise NotImplementedError('_gen_grid needs to be implemented by each environment')

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        return 1 - .9 * (self.step_count / self.max_steps)

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def reset(self):
        self.agent.reset()
        self._gen_grid(self.height, self.width)

        assert self.agent.pos is not None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.get_obs()
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.rng, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def __getitem__(self, pos):
        if not isinstance(pos[0], slice) and not isinstance(pos[1], slice):
            return self.grid[pos]

        env = copy.copy(self)
        env.agent = copy.copy(self.agent)
        env.height = pos[0].stop - pos[0].start
        env.width = pos[1].stop - pos[1].start
        env.grid = Grid(*env.shape)
        from_, to_ = self._slices(pos)
        env.agent.pos = (self.agent.pos[0] - from_[0].start,
                         self.agent.pos[1] - from_[1].start)
        env.grid[to_] = self.grid[from_]
        return env

    def __setitem__(self, pos, obj):
        if isinstance(obj, MiniGridEnv):
            from_, to_ = self._slices(pos)
            # if not isinstance(pos[0], slice) and not isinstance(pos[1], slice):
            #     self._grid[pos] = obj
            # else:
            self.grid[from_] = obj.grid[to_]
        elif isinstance(obj, Cell):
            self.grid[pos] = obj
        else:
            self[pos].entity = obj
            obj.pos = pos

    def _get_from_to_slices(self, slice_, axis):
        top = slice_.start
        bottom = slice_.stop
        offset = -top if top < 0 else 0
        top = max(0, top)
        bottom = min(self.shape[axis], bottom)
        size = bottom - top
        return slice(top, top + size), slice(offset, offset + size)

    def _slices(self, pos):
        if isinstance(pos[0], slice):
            from_i, to_i = self._get_from_to_slices(pos[0], axis=0)
        else:
            from_i = slice(pos[0], pos[0] + 1)

        if isinstance(pos[1], slice):
            from_j, to_j = self._get_from_to_slices(pos[1], axis=1)
        else:
            from_j = slice(pos[1], pos[1] + 1)

        return (from_i, from_j), (to_i, to_j)

    def encode(self, mask=None, as_str=False):
        """
        Produce a compact numpy encoding of the grid
        """
        if mask is None:
            mask = np.ones(self.shape, dtype=bool)

        n_attrs = len(self[0,0].to_array())
        assert n_attrs == 3
        array = np.zeros((n_attrs,) + self.shape)
        if as_str:
            array = array.astype(np.object)
        else:
            array = array.astype('uint8')

        for (i, j), cell in np.ndenumerate(self.grid):
            if as_str:
                array[:, i, j] = cell.to_array()  # type, color, state
            else:
                array[:, i, j] = cell.to_idx_array()
        # print('$' * 10)
        # print('encode', array)
        # print('$' * 10)
        return array

    # @staticmethod
    def decode(self, array):
        """
        Decode an observation back into a grid
        """
        channels, height, width = array.shape
        assert channels == 3
        assert height == width

        # breakpoint()
        env = copy.copy(self)
        env.agent = copy.copy(self.agent)
        env.height = height
        env.width = width
        env.grid = Grid(height, width)
        env.agent.pos = (height, width // 2)
        env.agent.state = 'up'

        for (i, j), _ in np.ndenumerate(env.grid):
            type_, color, state = array[:, i, j]
            if type_ != 0:
                obj = entities.make(entities.OBJECTS[type_ - 1],
                                    color=entities.COLORS[color])
                if hasattr(obj, 'STATES'):
                    obj.state = obj.STATES[state]
                env[i, j].entity = obj

        return env

    def visible(self):
        """
        Process occluders and visibility
        Note that this incurs some performance cost  # TODO: verify if still true
        """
        def _flood_fill(pos):
            if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.shape[0] or pos[1] >= self.shape[1]:
                return

            if mask[pos]:  # already visited
                return

            if view_box[pos] == 0:  # outside agent's view_size
                return

            if self[pos].entity is not None:
                if not self[pos].entity.see_behind():   # this is a boundary
                    mask[pos] = True  # add it to the list of visibles and exit
                    return

            mask[pos] = True

            # visit all neighbors in the surrounding square
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i != 0 or j != 0:  # skip current pos
                        _flood_fill((pos[0] + i, pos[1] + j))

        view_box = np.zeros(self.shape, dtype=bool)
        from_, to_ = self._slices(self.agent.view_box)
        view_box[from_[0], from_[1]] = True

        if self.see_through_walls:
            mask = np.ones(self.shape, dtype=bool)
        else:
            mask = np.zeros(self.shape, dtype=bool)
            _flood_fill(self.agent.pos)
        return mask

    def visited(self):
        arr = np.zeros(self.shape, dtype=bool)
        for pos in self.agent.visited:
            arr[pos] = True
        return arr

    def place_obj(self,
                  obj,
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

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = (
                self.rng.randint(top[0], min(top[0] + size[0], self.height)),
                self.rng.randint(top[1], min(top[1] + size[1], self.width))
            )

            # Don't place the object on top of another object
            # TODO: may want to consider can_overlap and can_contain cases
            if self[pos].entity is not None:
                continue

            # Check if there is a filtering criterion
            if reject_fn is not None and reject_fn(self, pos):
                continue

            break

        if obj.type == 'agent':
            obj.pos = pos
        else:
            self[pos] = obj

    def place_agent(self, top=(0,0), size=None, rand_dir=True,
                    max_tries=math.inf):
        """
        Set the agent's starting point at an empty position in the grid
        """
        self.place_obj(self.agent, top=top, size=size, max_tries=max_tries)
        if rand_dir:
            self.agent.state = self.rng.choice(self.agent.STATES)

    def horz_wall(self, i, j, width=None, obj=entities.Wall):
        if width is None:
            width = self.width - j
        for jj in range(0, width):
            self[i, j + jj] = obj()

    def vert_wall(self, i, j, height=None, obj=entities.Wall):
        if height is None:
            height = self.height - i
        for ii in range(0, height):
            self[i + ii, j] = obj()

    def wall_rect(self, i, j, height, width):
        self.horz_wall(i, j, width)
        self.horz_wall(i + height - 1, j, width)
        self.vert_wall(i, j, height)
        self.vert_wall(i, j + width - 1, height)

    def move_agent(self, pos):
        cell = self[pos]
        if cell.entity is None or cell.entity.can_overlap():
            self.agent.pos = pos

    def pickup(self, pos):
        cell = self[pos]
        if cell.entity is not None:
            if cell.entity.can_pickup() and not self.agent.is_carrying:
                self.agent.carrying = cell.entity
                self[pos].clear()

    def drop(self, pos):
        cell = self[pos]
        if cell.entity is None and self.agent.is_carrying:
            self[pos] = self.agent.carrying
            self.agent.carrying = None

    def step(self, action):
        self.step_count += 1
        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.agent.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self[fwd_pos]

        action = self.actions[action]

        # Rotate left
        if action == 'left':
            self.agent.rotate_left()

        # Rotate right
        elif action == 'right':
            self.agent.rotate_right()

        # Move forward
        elif action == 'forward':
            self.move_agent(fwd_pos)
            if fwd_cell.entity is not None:
                if fwd_cell.entity.type == 'goal':
                    done = True
                    reward = self._reward()
                if fwd_cell.entity.type == 'lava':
                    done = True

        # Pick up an object
        elif action == 'pickup':
            self.pickup(fwd_pos)

        # Drop an object
        elif action == 'drop':
            self.drop(fwd_pos)

        # Toggle/activate an object
        elif action == 'toggle':
            if fwd_cell.entity is not None:
                fwd_cell.entity.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == 'done':
            pass

        else:
            raise ValueError(f'unknown action {action}')

        if self.step_count >= self.max_steps:
            done = True

        obs = self.get_obs()

        return obs, reward, done, {}

    def get_obs(self, as_str=False):
        """
        Generate the agent's view (partially observable,
        low-resolution encoding)
        """
        if not hasattr(self, 'mission'):
            raise AttributeError('environments must define a textual mission string')

        # take a slice of the environment within agent's view size
        obs = self[self.agent.view_box]

        im = obs.encode(as_str=as_str)
        # remove the parts of environment that the agent cannot see
        im[:, ~obs.visible()] = None if as_str else 0

        # orient agent upright
        im = np.rot90(im, k=self.agent.dir + 1, axes=(1,2))

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        # print(im)
        obs = {'image': im,
               'direction': self.agent.state,
               'mission': self.mission
               }
        return obs

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
            return render_text.ANSI(self)()
        elif mode == 'curses4bit':
            return render_text.Curses4bit(self)()
        elif mode == 'curses8bit':
            return render_text.Curses8bit(self)()
        else:
            raise ValueError(f'Render mode {mode} not recognized')
