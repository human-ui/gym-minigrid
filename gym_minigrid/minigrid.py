import math, copy, enum

import numpy as np
import gym

from gym_minigrid import entities, render_text


class Cell(object):

    def __init__(self, color='black'):
        self._orig_color = color
        self.clear()

    def clear(self):
        self.color = self._orig_color
        # self.visible = False
        # self.visited = False
        self.entity = None

    def __str__(self):

        if self.entity is not None:
            r = f', has:\n  {self.entity}'
        else:
            r = ', empty'
        return f'cell: {self.color}, visible: {self.visible}, visited: {self.visited}{r}'

    def copy(self):
        return copy.deepcopy(self)

    def to_array(self):
        if self.entity is None:
            entity = np.array([None, None, None])
        else:
            entity = self.entity.to_array()

        arr = np.concatenate([
            entity,
            np.array([self.color])  # , None, None])
        ])
        return arr

    def to_idx_array(self):
        if self.entity is None:
            entity = np.array([0, 0, 0])
        else:
            entity = self.entity.to_idx_array()

        arr = np.concatenate([
            entity,
            np.array([
                entities.COLORS.index(self.color),
                # 0,
                # 0
            ])
        ])
        return arr


class Grid(object):
    """
    Represent a grid and operations on it

    It is defined as an array of Cells.
    """

    def __init__(self, height, width):
        if height < 3:
            raise ValueError(f'minigrid height must be larger than 3, got {height}')
        if width < 3:
            raise ValueError(f'minigrid widht must be larger than 3, got {width}')

        self._grid = np.empty((height, width), dtype=np.object)
        for (i, j), _ in np.ndenumerate(self._grid):
            self._grid[i, j] = Cell()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif f'_{attr}' in self.__dict__:
            return self.__dict__[f'_{attr}']
        elif '_grid' in self.__dict__:
            return getattr(self._grid, attr)
        else:
            raise AttributeError(f'No attribute {attr}')

    def __getitem__(self, pos):
        item = self._grid[tuple(pos)]
        if isinstance(item, np.ndarray):  # if slicing
            g = Grid(3, 3)
            g._grid = item
            return g
        else:
            return item

    def __setitem__(self, pos, obj):
        pos = tuple(pos)
        if obj is None:
            self._grid[pos].clear()
        else:
            self._grid[pos].entity = obj
            obj.pos = pos

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

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    @property
    def shape(self):
        return self._grid.shape

    def copy(self):
        return copy.deepcopy(self)

    def encode(self, mask=None, as_str=False):
        """
        Produce a compact numpy encoding of the grid
        """
        if mask is None:
            mask = np.ones(self._grid.shape, dtype=bool)

        n_attrs = len(self._grid[0,0].to_array())
        array = np.zeros((n_attrs,) + self._grid.shape)
        if as_str:
            array = array.astype(np.object)
        else:
            array = array.astype('uint8')

        for (i, j), cell in np.ndenumerate(self._grid):

            if as_str:
                if mask[i, j]:
                    array[:, i, j] = cell.to_array()
                    # array[4, i, j] = True
                else:
                    array[:, i, j] = None
            else:
                if mask[i, j]:
                    array[:, i, j] = cell.to_idx_array()
                    # array[4, i, j] = 1
                else:
                    array[:, i, j] = 0

        return array

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        channels, height, width = array.shape
        assert channels == 4

        grid = Grid(height, width)
        for i in range(height):
            for j in range(width):
                type_, color, state, bckg_color = array[:, i, j]
                grid[i, j].color = entities.COLORS[bckg_color]
                # grid[i, j].visible = visible
                # grid[i, j].visited = visited
                obj = entities.make(entities.OBJECTS[type_], color=entities.COLORS[color])
                if hasattr(obj, 'STATES'):
                    obj.state = obj.STATES[state]
                grid[i, j].entity = obj

        return grid


class ObsGrid(Grid):

    def __init__(self, grid, agent, see_through_walls=False, orient_up=True):
        self.see_through_walls = see_through_walls

        self.agent = copy.copy(agent)
        top_i, top_j, bottom_i, bottom_j = self.agent.view_box
        offset_i = -top_i if top_i < 0 else 0
        offset_j = -top_j if top_j < 0 else 0
        top_i = max(0, top_i)
        top_j = max(0, top_j)

        grid = grid[top_i: bottom_i, top_j: bottom_j]

        self._grid = np.empty((self.agent.view_size, self.agent.view_size), dtype=np.object)
        for pos, _ in np.ndenumerate(self._grid):
            if pos[0] - offset_i < 0 or pos[1] - offset_j < 0 or pos[0] - offset_i >= grid.shape[0] or pos[1] - offset_j >= grid.shape[1]:
                self._grid[pos] = Cell()
            else:
                cell = grid[pos[0] - offset_i, pos[1] - offset_j]
                self._grid[pos] = cell.copy()

        self.agent.pos = (self.agent.pos[0] + offset_i - top_i,
                          self.agent.pos[1] + offset_j - top_j)

        # orient the grid in such a way that the agent is always
        # at the bottom and looking up
        if orient_up:
            for _ in range(self.agent.dir + 1):
                self.rotate_left()

    def mask(self):
        if self.see_through_walls:
            self._mask = np.ones(self.shape, dtype=bool)
        else:
            self._mask = np.zeros(self.shape, dtype=bool)
            self._flood_fill(self.agent.pos)
        return self._mask

    def _flood_fill(self, pos):
        if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.shape[0] or pos[1] >= self.shape[1]:
            return

        if self._mask[pos]:  # already visited
            return
        if self[pos].entity is not None:
            if not self[pos].entity.see_behind():   # this is a boundary
                self._mask[pos] = True  # add it to the list of visibles and exit
                return

        self._mask[pos] = True

        # visit all neighbors in the surrounding square
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i != 0 or j != 0:  # skip current pos
                    self._flood_fill((pos[0] + i, pos[1] + j))

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """
        self._grid = np.rot90(self._grid)
        self.agent.pos = (self.shape[0] - self.agent.pos[1], self.agent.pos[0])
        self.agent.rotate_left()
        return self

    def encode(self):
        return super().encode(self.mask())


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
        self.actions = self.agent.ACTIONS

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

    def _gen_grid(self, height, width):
        raise NotImplementedError('_gen_grid needs to be implemented by each environment')

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        return 1 - .9 * (self.step_count / self.max_steps)

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def mask(self):
        """
        Process occluders and visibility
        Note that this incurs some performance cost
        """
        mask = np.zeros(self.grid.shape, dtype=bool)
        obsgrid = self.get_obsgrid(orient_up=False)
        top_i, top_j, bottom_i, bottom_j = self.agent.view_box

        offset_i = -top_i if top_i < 0 else 0
        offset_j = -top_j if top_j < 0 else 0
        top_i = max(0, top_i)
        top_j = max(0, top_j)
        bottom_i = min(self.height, bottom_i)
        bottom_j = min(self.width, bottom_j)

        obsgrid_mask = obsgrid.mask()
        mask[top_i: bottom_i, top_j: bottom_j] = obsgrid_mask[offset_i: offset_i + bottom_i - top_i, offset_j: offset_j + bottom_j - top_j]
        return mask

    def reset(self):
        self.last_action = None
        self.agent.reset()

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.height, self.width)

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.get_obs()
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.rng, seed = gym.utils.seeding.np_random(seed)
        return [seed]

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
            if self.grid[pos].entity is not None:
                continue

            # Check if there is a filtering criterion
            if reject_fn is not None and reject_fn(self, pos):
                continue

            break

        if obj.type == 'agent':
            obj.pos = pos
        else:
            self.grid[pos] = obj

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
            self.grid[i, j + jj] = obj()

    def vert_wall(self, i, j, height=None, obj=entities.Wall):
        if height is None:
            height = self.height - i
        for ii in range(0, height):
            self.grid[i + ii, j] = obj()

    def wall_rect(self, i, j, height, width):
        self.horz_wall(i, j, width)
        self.horz_wall(i + height - 1, j, width)
        self.vert_wall(i, j, height)
        self.vert_wall(i, j + width - 1, height)

    def move_agent(self, pos):
        cell = self.grid[pos]
        if cell.entity is None or cell.entity.can_overlap():
            self.agent.pos = pos

    def pickup(self, pos):
        cell = self.grid[pos]
        if cell.entity is not None:
            if cell.entity.can_pickup() and not self.agent.is_carrying:
                self.agent.carrying = cell.entity
                self.grid[pos].clear()

    def drop(self, pos):
        cell = self.grid[pos]
        if cell.entity is None and self.agent.is_carrying:
            self.grid[pos] = self.agent.carrying
            self.agent.carrying = None

    def step(self, action):
        self.step_count += 1
        self.last_action = action
        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.agent.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid[fwd_pos]

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
            raise ValueError('unknown action')

        if self.step_count >= self.max_steps:
            done = True

        obs = self.get_obs()

        return obs, reward, done, {}

    def get_obs(self, encoded=True, orient_up=True):
        """
        Generate the agent's view (partially observable,
        low-resolution encoding)
        """
        if not hasattr(self, 'mission'):
            raise AttributeError('environments must define a textual mission string')

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obsgrid = self.get_obsgrid(orient_up=orient_up)
        obs = {'image': obsgrid.encode(),
               'direction': obsgrid.agent.state,
               'mission': self.mission
               }
        return obs

    def get_obsgrid(self, orient_up=True):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """
        return ObsGrid(self.grid, self.agent, see_through_walls=self.see_through_walls, orient_up=orient_up)

    def render(self, mode='human', highlight=True, cell_pixels=32):
        if mode in ['human', 'rgb_array', 'pixmap']:
            from gym_minigrid.render_image import MiniGridImage
            im = MiniGridImage(self, highlight=highlight, cell_pixels=cell_pixels)
            if mode == 'human':
                im.show()
            elif mode == 'rgb_array':
                return im.array()
            elif mode == 'pixmap':
                return im.pixmap()

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
