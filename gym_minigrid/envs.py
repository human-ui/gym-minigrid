import sys, itertools, operator, inspect
import numpy as np
import gym

from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid import encoding
from gym_minigrid.encoding import ATTRS

CH = encoding.Channels()


class Empty(MiniGridEnv):
    """
    This environment is an empty room, and the goal of the agent is to reach the green goal square, which provides a sparse reward. A small penalty is subtracted for the number of steps to reach the goal. This environment is useful, with small rooms, to validate that your RL algorithm works correctly, and with large rooms to experiment with sparse rewards and exploration. The random variants of the environment have the agent starting at a random position for each episode, while the regular variants have the agent always starting in the corner opposite to the goal.
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_state='right',
        max_steps=None,
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_state = agent_start_state
        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            height=size,
            width=size,
            max_steps=max_steps,
            **kwargs
        )

    def _gen_grid(self):
        # Generate the surrounding walls
        self.wall_rect(0, 0, self.height, self.width)

        # Place a goal square in the bottom-right corner
        self.set_obj((self.height - 2, self.width - 2), 'goal')

        # Place the agent
        self.set_attr(self.agent_start_pos, 'agent_pos')
        self.set_attr(self.agent_pos, 'agent_state', self.agent_start_state)

        self.mission = 'get to the green goal square'


class FourRooms(MiniGridEnv):
    """
    Classic four room reinforcement learning environment. The agent must navigate in a maze composed of four rooms interconnected by 4 gaps in the walls. To obtain a reward, the agent must reach the green goal square. Both the agent and the goal square are randomly placed in any of the four rooms.
    """

    def __init__(self, size=19, agent_pos=None, goal_pos=None, max_steps=100, **kwargs):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(height=size, width=size, max_steps=max_steps, **kwargs)

    def _gen_grid(self):
        # Generate the surrounding walls
        self.horz_wall(0, 0)
        self.horz_wall(self.height - 1, 0)
        self.vert_wall(0, 0)
        self.vert_wall(0, self.width - 1)

        room_w = self.width // 2
        room_h = self.height // 2

        # For each row of rooms
        for i in range(0, 2):

            # For each column
            for j in range(0, 2):
                i_top = i * room_h
                j_left = j * room_w
                i_bottom = i_top + room_h
                j_right = j_left + room_w

                # Right wall and door
                if j + 1 < 2:
                    self.vert_wall(i_top, j_right, room_h)
                    pos = (self.rng.randint(i_top + 1, i_bottom), j_right)
                    self.clear(pos)

                # Bottom wall and door
                if i + 1 < 2:
                    self.horz_wall(i_bottom, j_left, room_w)
                    pos = (i_bottom, self.rng.randint(j_left + 1, j_right))
                    self.clear(pos)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.set_attr(self._agent_default_pos, 'agent_pos')
            state = self.rng.choice(CH.attrs['agent_state'])
            self.set_attr(self.agent_pos, 'agent_state', state)
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            self.set_obj(self._goal_default_pos, 'goal')
        else:
            self.place_obj('goal')

        self.mission = 'Reach the goal'


class MultiRoom(MiniGridEnv):
    """
    This environment has a series of connected rooms with doors that must be opened in order to get to the next room. The final room has the green goal square the agent must get to. This environment is extremely difficult to solve using RL alone. However, by gradually increasing the number of rooms and building a curriculum, the environment can be solved.
    """

    def __init__(self,
                 n_rooms=6,
                 max_room_size=10,
                 max_steps=None,
                 **kwargs
                 ):
        assert n_rooms > 0
        assert max_room_size >= 4

        self.n_rooms = n_rooms
        self.max_room_size = max_room_size
        if max_steps is None:
            max_steps = self.n_rooms * 20

        height = max_room_size * int(np.ceil(np.sqrt(n_rooms)))
        width = height

        self.rooms = []

        super().__init__(
            height=height,  # TODO
            width=width,
            max_steps=max_steps,
            **kwargs
        )

    def _gen_grid(self):
        room_list = []

        while len(room_list) < self.n_rooms:
            cur_room_list = []

            entry_door_pos = (
                self.rng.randint(0, self.height - 2),
                self.rng.randint(0, self.width - 2)
            )

            # Recursively place the rooms
            self._place_room(
                self.n_rooms,
                room_list=cur_room_list,
                min_sz=4,
                max_sz=self.max_room_size,
                entry_door_wall=2,
                entry_door_pos=entry_door_pos
            )

            if len(cur_room_list) > len(room_list):
                room_list = cur_room_list

        # Store the list of rooms in this environment
        assert len(room_list) > 0
        self.rooms = room_list

        prev_door_color = None

        # For each room
        for idx, room in enumerate(room_list):

            top_i, top_j = room['top']
            room_height, room_width = room['size']

            # Generate the surrounding walls
            self.horz_wall(top_i, top_j, width=room_width)
            self.horz_wall(top_i + room_height - 1, top_j, width=room_width)
            self.vert_wall(top_i, top_j, height=room_height)
            self.vert_wall(top_i, top_j + room_width - 1, height=room_height)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                door_colors = list(CH.attrs['object_color'])
                if prev_door_color:
                    door_colors.remove(prev_door_color)
                door_colors = self.rng.choice(door_colors)

                self.set_obj(room['entry_door_pos'], 'door', color=door_colors, state='closed')
                prev_door_color = door_colors

                prev_room = room_list[idx - 1]
                prev_room['exit_door_pos'] = room['entry_door_pos']

        # Randomize the starting agent position and direction
        self.place_agent(top=room_list[0]['top'], size=room_list[0]['size'])

        # Place the final goal in the last room
        self.place_obj('goal', top=room_list[-1]['top'], size=room_list[-1]['size'])

        self.mission = 'traverse the rooms to get to the goal'

    def _place_room(
        self,
        num_left,
        room_list,
        min_sz,
        max_sz,
        entry_door_wall,
        entry_door_pos
    ):
        # Choose the room size randomly
        size_i = self.rng.randint(min_sz, max_sz + 1)
        size_j = self.rng.randint(min_sz, max_sz + 1)

        # The first room will be at the door position
        if len(room_list) == 0:
            top_i, top_j = entry_door_pos
        # Entry on the right
        elif entry_door_wall == 0:
            i = entry_door_pos[0]
            top_i = self.rng.randint(i - size_i + 2, i)
            top_j = entry_door_pos[1] - size_j + 1
        # Entry wall on the bottom
        elif entry_door_wall == 1:
            top_i = entry_door_pos[0] - size_i + 1
            j = entry_door_pos[1]
            top_j = self.rng.randint(j - size_j + 2, j)
        # Entry wall on the left
        elif entry_door_wall == 2:
            i = entry_door_pos[0]
            top_i = self.rng.randint(i - size_i + 2, i)
            top_j = entry_door_pos[1]
        # Entry wall on the top
        elif entry_door_wall == 3:
            top_i = entry_door_pos[0]
            j = entry_door_pos[1]
            top_j = self.rng.randint(j - size_j + 2, j)
        else:
            raise ValueError(f'Entry door wall index wrong: {entry_door_wall}')

        # If the room is out of the grid, can't place a room here
        if top_i < 0 or top_j < 0:
            return False
        if top_i + size_i >= self.height or top_j + size_j > self.width:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in room_list[:-1]:
            non_overlap = (top_i + size_i < room['top'][0]
                           or room['top'][0] + room['size'][0] <= top_i
                           or top_j + size_j < room['top'][1]
                           or room['top'][1] + room['size'][1] <= top_j)

            if not non_overlap:
                return False

        # Add this room to the list
        this_room = dict(
            top=(top_i, top_j),
            size=(size_i, size_j),
            entry_door_pos=entry_door_pos,
            exit_door_pos=None
        )
        room_list.append(this_room)

        # If this was the last room, stop
        if num_left == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wall_set = set((0, 1, 2, 3))
            wall_set.remove(entry_door_wall)
            exit_door_wall = self.rng.choice(sorted(wall_set))
            next_entry_wall = (exit_door_wall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exit_door_wall == 0:
                exit_door_pos = (
                    top_i + self.rng.randint(1, size_i - 1),
                    top_j + size_j - 1
                )
            # Exit on bottom wall
            elif exit_door_wall == 1:
                exit_door_pos = (
                    top_i + size_i - 1,
                    top_j + self.rng.randint(1, size_j - 1)
                )
            # Exit on left wall
            elif exit_door_wall == 2:
                exit_door_pos = (
                    top_i + self.rng.randint(1, size_i - 1),
                    top_j
                )
            # Exit on top wall
            elif exit_door_wall == 3:
                exit_door_pos = (
                    top_i,
                    top_j + self.rng.randint(1, size_j - 1)
                )
            else:
                raise ValueError

            # Recursively create the other rooms
            success = self._place_room(
                num_left - 1,
                room_list=room_list,
                min_sz=min_sz,
                max_sz=max_sz,
                entry_door_wall=next_entry_wall,
                entry_door_pos=exit_door_pos
            )

            if success:
                break

        return True


class RandomObjects(MiniGridEnv):
    """
    This environment is a blank grid filled with randomly placed objects (including wall elements). Useful for curriculum learning as the first learning stage.
    """

    def __init__(self,
                 size=16,
                 density=.2,
                 objects=CH.attrs['object_type'],
                 colors=CH.attrs['object_color'],
                 max_steps=100,
                 surround_walls=True,
                 **kwargs):
        self.density = density
        self.objects = [o for o in objects if o != 'goal']
        self.colors = colors
        self.in_box_objects = CH.attrs['carrying_type']
        self.in_box_colors = CH.attrs['carrying_color']
        self.surround_walls = surround_walls
        super().__init__(height=size, width=size, max_steps=max_steps, **kwargs)

    def _gen_grid(self):
        # Generate the surrounding walls
        if self.surround_walls:
            self.horz_wall(0, 0)
            self.horz_wall(self.height - 1, 0)
            self.vert_wall(0, 0)
            self.vert_wall(0, self.width - 1)

        # Place a goal square at a random location
        self.place_obj('goal')

        # Place random objects in the world
        if self.surround_walls:
            n_objs = int((self.height - 2) * (self.width - 2) * self.density)
        else:
            n_objs = int(self.height * self.width * self.density)

        for i in range(n_objs):
            self.make_obj()

        # Randomize the player start position and orientation
        agent_pos = (self.height // 2, self.width // 2)
        self.clear(agent_pos)
        self.set_attr(agent_pos, 'agent_pos')
        state = self.rng.choice(CH.attrs['agent_state'])
        self.set_attr(agent_pos, 'agent_state', state)
        self.mission = ''

    def make_obj(self):
        type_ = self.rng.choice(self.objects)
        color = self.rng.choice(self.colors)
        if type_ == 'door':
            state = self.rng.choice(CH.attrs['door_state'])
        else:
            state = None

        pos = self.place_obj(type_, color=color, state=state)

        if type_ == 'box':
            if self.rng.random() < .5:
                in_box_type = self.rng.choice(self.in_box_objects)
                in_box_color = self.rng.choice(self.in_box_colors)
                self.set_carrying_obj(pos, in_box_type, color=in_box_color)


# Register all environments with OpenAI gym
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__ and not name.startswith('_'):

        gym.envs.registration.register(
            id=f'MiniGrid-{name}-v2',
            entry_point=f'gym_minigrid.envs:{name}',
            reward_threshold=.95,
        )
