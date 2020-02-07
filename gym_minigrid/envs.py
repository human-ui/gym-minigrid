import sys, itertools, operator, inspect
import numpy as np
import gym

from gym_minigrid.minigrid import Grid, MiniGridEnv
from gym_minigrid.roomgrid import RoomGrid
import gym_minigrid.entities as entities
from gym_minigrid.entities import Goal, Wall, Door, Key, Ball, Box, Lava, COLORS, OBJECTS


class Empty(MiniGridEnv):
    """
    This environment is an empty room, and the goal of the agent is to reach the green goal square, which provides a sparse reward. A small penalty is subtracted for the number of steps to reach the goal. This environment is useful, with small rooms, to validate that your RL algorithm works correctly, and with large rooms to experiment with sparse rewards and exploration. The random variants of the environment have the agent starting at a random position for each episode, while the regular variants have the agent always starting in the corner opposite to the goal.
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_state='right',
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_state = agent_start_state

        super().__init__(
            height=size,
            width=size,
            max_steps=4 * size**2,
            **kwargs
        )

    def _gen_grid(self, height, width):
        # Create an empty grid
        self.grid = Grid(height, width)

        # Generate the surrounding walls
        self.wall_rect(0, 0, height, width)

        # Place a goal square in the bottom-right corner
        self[height - 2, width - 2] = Goal()

        # Place the agent
        self.agent.pos = self.agent_start_pos
        self.agent.state = self.agent_start_state

        self.mission = 'get to the green goal square'


class FourRooms(MiniGridEnv):
    """
    Classic four room reinforcement learning environment. The agent must navigate in a maze composed of four rooms interconnected by 4 gaps in the walls. To obtain a reward, the agent must reach the green goal square. Both the agent and the goal square are randomly placed in any of the four rooms.
    """

    def __init__(self, agent_pos=None, goal_pos=None, **kwargs):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(height=19, width=19, max_steps=100, **kwargs)

    def _gen_grid(self, height, width):
        # Create the grid
        self.grid = Grid(height, width)

        # Generate the surrounding walls
        self.horz_wall(0, 0)
        self.horz_wall(height - 1, 0)
        self.vert_wall(0, 0)
        self.vert_wall(0, width - 1)

        room_w = width // 2
        room_h = height // 2

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
                    self[pos].clear()

                # Bottom wall and door
                if i + 1 < 2:
                    self.horz_wall(i_bottom, j_left, room_w)
                    pos = (i_bottom, self.rng.randint(j_left + 1, j_right))
                    self[pos].clear()

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent.pos = self._agent_default_pos
            self.agent.state = self.rng.choice(self.agent.STATES)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            self[self._goal_default_pos] = Goal()
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'


class DoorKey(MiniGridEnv):
    """
    This environment has a key that the agent must pick up in order to unlock a goal and then get to the green goal square. This environment is difficult, because of the sparse reward, to solve using classical RL algorithms. It is useful to experiment with curiosity or curriculum learning.
    """

    def __init__(self, size=8, **kwargs):
        super().__init__(
            height=size,
            width=size,
            max_steps=10 * size * size,
            **kwargs
        )

    def _gen_grid(self, height, width):
        # Create an empty grid
        self.grid = Grid(height, width)

        # Generate the surrounding walls
        self.wall_rect(0, 0, height, width)

        # Place a goal in the bottom-right corner
        self[height - 2, width - 2] = Goal()

        # Create a vertical splitting wall
        split_idx = self.rng.randint(2, width - 2)
        self.vert_wall(0, split_idx)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(height, split_idx))

        # Place a door in the wall
        door_idx = self.rng.randint(1, height - 2)
        self[door_idx, split_idx] = Door('yellow', state='locked')

        # Place a yellow key on the left side
        self.place_obj(Key('yellow'), top=(0, 0), size=(height, split_idx))

        self.mission = 'use the key to open the door and then get to the goal'


class _MultiRoom(object):

    def __init__(self,
                 top,
                 size,
                 entry_door_pos,
                 exit_door_pos
                 ):
        self.top = top
        self.size = size
        self.entry_door_pos = entry_door_pos
        self.exit_door_pos = exit_door_pos


class MultiRoom(MiniGridEnv):
    """
    This environment has a series of connected rooms with doors that must be opened in order to get to the next room. The final room has the green goal square the agent must get to. This environment is extremely difficult to solve using RL alone. However, by gradually increasing the number of rooms and building a curriculum, the environment can be solved.
    """

    def __init__(self,
                 min_num_rooms=6,
                 max_num_rooms=6,
                 max_room_size=10,
                 **kwargs
                 ):
        assert min_num_rooms > 0
        assert max_num_rooms >= min_num_rooms
        assert max_room_size >= 4

        self.min_num_rooms = min_num_rooms
        self.max_num_rooms = max_num_rooms
        self.max_room_size = max_room_size

        self.rooms = []

        super().__init__(
            height=25,
            width=25,
            max_steps=self.max_num_rooms * 20,
            **kwargs
        )

    def _gen_grid(self, height, width):
        room_list = []

        # Choose a random number of rooms to generate
        num_rooms = self.rng.randint(self.min_num_rooms, self.max_num_rooms + 1)

        while len(room_list) < num_rooms:
            cur_room_list = []

            entry_door_pos = (
                self.rng.randint(0, height - 2),
                self.rng.randint(0, width - 2)
            )

            # Recursively place the rooms
            self._place_room(
                num_rooms,
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

        # Create the grid
        self.grid = Grid(height, width)

        prev_door_color = None

        # For each room
        for idx, room in enumerate(room_list):

            top_i, top_j = room.top
            room_height, room_width = room.size

            # Generate the surrounding walls
            self.horz_wall(top_i, top_j, width=room_width)
            self.horz_wall(top_i + room_height - 1, top_j, width=room_width)
            self.vert_wall(top_i, top_j, height=room_height)
            self.vert_wall(top_i, top_j + room_width - 1, height=room_height)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                door_colors = set(COLORS)
                if prev_door_color:
                    door_colors.remove(prev_door_color)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                door_colors = self.rng.choice(sorted(door_colors))

                self[room.entry_door_pos] = Door(door_colors)
                prev_door_color = door_colors

                prev_room = room_list[idx - 1]
                prev_room.exit_door_pos = room.entry_door_pos

        # Randomize the starting agent position and direction
        self.place_agent(room_list[0].top, room_list[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), room_list[-1].top, room_list[-1].size)

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
            non_overlap = \
                top_i + size_i < room.top[0] or \
                room.top[0] + room.size[0] <= top_i or \
                top_j + size_j < room.top[1] or \
                room.top[1] + room.size[1] <= top_j

            if not non_overlap:
                return False

        # Add this room to the list
        room_list.append(_MultiRoom(
            (top_i, top_j),
            (size_i, size_j),
            entry_door_pos,
            None
        ))

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


class Fetch(MiniGridEnv):
    """
    This environment has multiple objects of assorted types and colors. The agent receives a textual string as part of its observation telling it which object to pick up. Picking up the wrong object produces a negative reward.
    """

    def __init__(
        self,
        size=8,
        num_objs=3,
        **kwargs
    ):
        self.num_objs = num_objs

        super().__init__(
            height=size,
            width=size,
            max_steps=5 * size**2,
            **kwargs
        )

    def _gen_grid(self, height, width):
        self.grid = Grid(height, width)

        # Generate the surrounding walls
        self.horz_wall(0, 0)
        self.horz_wall(height - 1, 0)
        self.vert_wall(0, 0)
        self.vert_wall(0, width - 1)

        types = ['key', 'ball']

        objs = []

        # For each object to be generated
        while len(objs) < self.num_objs:
            obj_type = self.rng.choice(types)
            obj_color = self.rng.choice(COLORS)

            if obj_type == 'key':
                obj = Key(obj_color)
            elif obj_type == 'ball':
                obj = Ball(obj_color)

            self.place_obj(obj)
            objs.append(obj)

        # Randomize the player start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        target = objs[self.rng.randint(0, len(objs))]
        self.target_type = target.type
        self.target_color = target.color

        # Generate the mission string
        missions = ['get a', 'go get a', 'fetch a', 'go fetch a', 'you must fetch a']
        self.mission = self.rng.choice(missions) + f' {self.target_color} {self.target_type}'

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.agent.is_carrying:
            if self.agent.carrying.color == self.target_color and \
               self.agent.carrying.type == self.target_type:
                reward = self._win_reward
            else:
                reward = self._lose_reward
            done = True

        return obs, reward, done, info


class GoToObject(MiniGridEnv):
    """
    This environment is a room with four doors, one on each wall. The agent receives a textual (mission) string as input, telling it which door to go to, (eg: "go to the red door"). It receives a positive reward for performing the `done` action next to the correct door, as indicated in the mission string. (BUG: doesn't look like the mission had that indicated)
    """

    def __init__(self, size=6, num_objs=2, **kwargs):
        self.num_objs = num_objs

        super().__init__(
            height=size,
            width=size,
            max_steps=5 * size**2,
            **kwargs
        )

    def _gen_grid(self, height, width):
        self.grid = Grid(height, width)

        # Generate the surrounding walls
        self.wall_rect(0, 0, height, width)

        # Types and colors of objects we can generate
        types = ['key', 'ball', 'box']

        objs = []
        # Until we have generated all the objects
        while len(objs) < self.num_objs:
            obj_type = self.rng.choice(types)
            obj_color = self.rng.choice(COLORS)

            # If this object already exists, try again
            if (obj_type, obj_color) in objs:
                continue

            if obj_type == 'key':
                obj = Key(obj_color)
            elif obj_type == 'ball':
                obj = Ball(obj_color)
            elif obj_type == 'box':
                obj = Box(obj_color)

            self.place_obj(obj)
            objs.append(obj)

        # Randomize the agent start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        self.target = self.rng.choice(objs)

        self.mission = f'go to the {self.target.color} {self.target.type}'

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # Toggle/pickup action terminates the episode
        if self.actions[action] == 'toggle':
            reward = self._lose_reward
            done = True

        # Reward performing the done action next to the target object
        ai, aj = self.agent.pos
        ti, tj = self.target.pos
        if self.actions[action] == 'done':
            reward = self._lose_reward
            if abs(ai - ti) <= 1 and abs(aj - tj) <= 1:
                reward = self._win_reward
            done = True

        return obs, reward, done, info


class GoToDoor(MiniGridEnv):
    """
    This environment is a room with four doors, one on each wall. The agent receives a textual (mission) string as input, telling it which door to go to, (eg: "go to the red door"). It receives a positive reward for performing the `done` action next to the correct door, as indicated in the mission string.
    """

    def __init__(self, size=5, **kwargs):
        assert size >= 5

        super().__init__(
            height=size,
            width=size,
            max_steps=5 * size**2,
            **kwargs
        )

    def _gen_grid(self, height, width):
        # Create the grid
        self.grid = Grid(height, width)

        # Randomly vary the room width and height
        height = self.rng.randint(5, height + 1)
        width = self.rng.randint(5, width + 1)

        # Generate the surrounding walls
        self.wall_rect(0, 0, height, width)

        # Generate the 4 doors at random positions
        door_pos = [(0, self.rng.randint(2, width - 2)),
                    (height - 1, self.rng.randint(2, width - 2)),
                    (self.rng.randint(2, height - 2), 0),
                    (self.rng.randint(2, height - 2), width - 1)]

        # Generate the door colors
        door_colors = self.rng.choice(COLORS, size=len(door_pos), replace=False)

        # Place the doors in the grid
        for idx, pos in enumerate(door_pos):
            color = door_colors[idx]
            self[pos] = Door(color)

        # Randomize the agent start position and orientation
        self.place_agent()

        # Select a random target door
        door_idx = self.rng.randint(0, len(door_pos))
        self.target_pos = door_pos[door_idx]
        self.target_color = door_colors[door_idx]

        # Generate the mission string
        self.mission = f'go to the {self.target_color} door'

    def step(self, action):
        obs, reward, done, info = super().step(action)

        ai, aj = self.agent.pos
        ti, tj = self.target_pos

        # Don't let the agent open any of the doors
        if self.actions[action] == 'toggle':
            reward = self._lose_reward
            done = True

        # Reward performing done action in front of the target door
        if self.actions[action] == 'done':
            reward = self._lose_reward
            if (ai == ti and abs(aj - tj) == 1) or (aj == tj and abs(ai - ti) == 1):
                reward = self._win_reward
            done = True

        return obs, reward, done, info


class PutNear(MiniGridEnv):
    """
    The agent is instructed through a textual string to pick up an object and place it next to another object. This environment is easy to solve with two objects, but difficult to solve with more, as it involves both textual understanding and spatial reasoning involving multiple objects.
    """

    def __init__(
        self,
        size=6,
        num_objs=2,
        **kwargs
    ):
        self.num_objs = num_objs

        super().__init__(
            height=size,
            width=size,
            max_steps=5 * size,
            **kwargs
        )

    def _gen_grid(self, height, width):
        self.grid = Grid(height, width)

        # Generate the surrounding walls
        self.horz_wall(0, 0)
        self.horz_wall(height - 1, 0)
        self.vert_wall(0, 0)
        self.vert_wall(0, width - 1)

        # Types and colors of objects we can generate
        types = ['key', 'ball', 'box']

        objs = []
        obj_pos = []

        def near_obj(env, p1):
            for p2 in obj_pos:
                di = p1[0] - p2[0]
                dj = p1[1] - p2[1]
                if abs(di) <= 1 and abs(dj) <= 1:
                    return True
            return False

        # Until we have generated all the objects
        while len(objs) < self.num_objs:
            obj_type = self.rng.choice(types)
            obj_color = self.rng.choice(COLORS)

            # If this object already exists, try again
            if (obj_type, obj_color) in objs:
                continue

            if obj_type == 'key':
                obj = Key(obj_color)
            elif obj_type == 'ball':
                obj = Ball(obj_color)
            elif obj_type == 'box':
                obj = Box(obj_color)

            self.place_obj(obj, reject_fn=near_obj)

            objs.append((obj_type, obj_color))
            obj_pos.append(obj.pos)

        # Randomize the agent start position and orientation
        self.place_agent()

        # Choose a random object to be moved
        obj_idx = self.rng.randint(0, len(objs))
        self.move_type, self.move_color = objs[obj_idx]
        # self.move_pos = obj_pos[obj_idx]

        # Choose a target object (to put the first object next to)
        while True:
            targetIdx = self.rng.randint(0, len(objs))
            if targetIdx != obj_idx:
                break
        self.target_type, self.target_color = objs[targetIdx]
        self.target_pos = obj_pos[targetIdx]

        self.mission = (f'put the {self.move_color} {self.move_type} near '
                        f'the {self.target_color} {self.target_type}')

    def step(self, action):
        pre_carrying = self.agent.carrying

        obs, reward, done, info = super().step(action)

        oi, oj = self.agent.front_pos
        ti, tj = self.target_pos

        # If we picked up the wrong object, terminate the episode
        if self.actions[action] == 'pickup' and self.agent.is_carrying:
            if self.agent.carrying.type != self.move_type or self.agent.carrying.color != self.move_color:
                reward = self._lose_reward
                done = True

        # If successfully dropping an object near the target
        if self.actions[action] == 'drop' and pre_carrying:
            reward = self._lose_reward
            if self[oj, oi] is pre_carrying:
                if abs(oi - ti) <= 1 and abs(oj - tj) <= 1:
                    reward = self._win_reward
            done = True

        return obs, reward, done, info


class RedBlueDoor(MiniGridEnv):
    """
    The purpose of this environment is to test memory. The agent is randomly placed within a room with one red and one blue door facing opposite directions. The agent has to open the red door and then open the blue door, in that order. The agent, when facing one door, cannot see the door behind him. Hence, the agent needs to remember whether or not he has previously opened the other door in order to reliably succeed at completing the task.
    """

    def __init__(self, size=8, **kwargs):
        self.size = size

        super().__init__(
            height=size,
            width=2 * size,
            max_steps=20 * size * size,
            **kwargs
        )

    def _gen_grid(self, height, width):
        # Create an empty grid
        self.grid = Grid(height, width)

        # Generate the grid walls
        self.wall_rect(0, 0, self.size, 2 * self.size)
        self.wall_rect(0, self.size // 2, self.size, self.size)

        # Place the agent in the top-left corner
        self.place_agent(top=(0, self.size // 2), size=(self.size, self.size))

        # Add a red door at a random position in the left wall
        pos = self.rng.randint(1, self.size - 1)
        self.red_door = Door('red')
        self[pos, self.size // 2] = self.red_door

        # Add a blue door at a random position in the right wall
        pos = self.rng.randint(1, self.size - 1)
        self.blue_door = Door('blue')
        self[pos, self.size // 2 + self.size - 1] = self.blue_door

        # Generate the mission string
        self.mission = 'open the red door then the blue door'

    def step(self, action):
        red_door_opened_before = self.red_door.is_open
        blue_door_opened_before = self.blue_door.is_open

        obs, reward, done, info = super().step(action)

        red_door_opened_after = self.red_door.is_open
        blue_door_opened_after = self.blue_door.is_open

        if blue_door_opened_after:
            if red_door_opened_before:
                reward = self._win_reward
                done = True
            else:
                reward = self._lose_reward
                done = True

        elif red_door_opened_after:
            if blue_door_opened_before:
                reward = self._lose_reward
                done = True

        return obs, reward, done, info


class Memory(MiniGridEnv):
    """
    This environment is a memory test. The agent starts in a small room
    where it sees an object. It then has to go through a narrow hallway
    which ends in a split. At each end of the split there is an object,
    one of which is the same as the object in the starting room. The
    agent has to remember the initial object, and go to the matching
    object at split.
    """

    def __init__(
        self,
        size=13,
        random_length=False,
        **kwargs
    ):
        self.random_length = random_length
        super().__init__(
            height=size,
            width=size,
            max_steps=5 * size**2,
            **kwargs
        )

    def _gen_grid(self, height, width):
        self.grid = Grid(height, width)

        # Generate the surrounding walls
        self.horz_wall(0, 0)
        self.horz_wall(height - 1, 0)
        self.vert_wall(0, 0)
        self.vert_wall(0, width - 1)

        assert height % 2 == 1
        upper_room_wall = height // 2 - 2
        lower_room_wall = height // 2 + 2
        if self.random_length:
            hallway_end = self.rng.randint(4, width - 2)
        else:
            hallway_end = width - 3

        # Start room
        self.horz_wall(upper_room_wall, 1, width=4)
        self.horz_wall(upper_room_wall + 1, 4, width=1)
        self.horz_wall(lower_room_wall, 1, width=4)
        self.horz_wall(lower_room_wall - 1, 4, width=1)

        # Horizontal hallway
        self.horz_wall(upper_room_wall + 1, 5, width=hallway_end - 5)
        self.horz_wall(lower_room_wall - 1, 5, width=hallway_end - 5)

        # Vertical hallway
        self.vert_wall(0, hallway_end, height=height)
        self.vert_wall(0, hallway_end + 2, height=height)
        self[height // 2, hallway_end].clear()

        # Fix the player's start position and orientation
        self.agent.pos = (height // 2,
                          self.rng.randint(1, hallway_end + 1))
        self.agent.state = 'right'

        # Place objects
        start_room_obj = self.rng.choice([Key, Ball])
        self[height // 2 - 1, 1] = start_room_obj('green')

        other_objs = self.rng.permutation([Ball, Key])
        pos0 = (height // 2 - 2, hallway_end + 1)
        pos1 = (height // 2 + 2, hallway_end + 1)
        self[pos0] = other_objs[0]('green')
        self[pos1] = other_objs[1]('green')

        # Choose the target objects
        if start_room_obj == other_objs[0]:
            self.success_pos = (pos0[0], pos0[1] + 1)
            self.failure_pos = (pos1[0], pos1[1] - 1)
        else:
            self.success_pos = (pos1[0], pos1[1] - 1)
            self.failure_pos = (pos0[0], pos0[1] + 1)

        self.mission = 'go to the matching object at the end of the hallway'

    def step(self, action):
        if self.actions[action] == 'pickup':
            action = self.actions.index('toggle')
        obs, reward, done, info = super().step(action)

        if tuple(self.agent.pos) == self.success_pos:
            reward = self._win_reward
            done = True
        if tuple(self.agent.pos) == self.failure_pos:
            reward = self._lose_reward
            done = True

        return obs, reward, done, info


class _LockedRoom(object):

    def __init__(self,
                 top,
                 size,
                 door_pos
                 ):
        self.top = top
        self.size = size
        self.door_pos = door_pos
        self.color = None
        self.locked = False

    def rand_pos(self, env):
        top_i, top_j = self.top
        size_i, size_j = self.size
        return (env.rng.randint(top_i + 1, top_i + size_i - 1),
                env.rng.randint(top_j + 1, top_j + size_j - 1))


class LockedRoom(MiniGridEnv):
    """
    The environment has six rooms, one of which is locked. The agent receives a textual mission string as input, telling it which room to go to in order to get the key that opens the locked room. It then has to go into the locked room in order to reach the final goal. This environment is extremely difficult to solve with vanilla reinforcement learning alone.
    """

    def __init__(self, size=19, **kwargs):
        super().__init__(height=size, width=size, max_steps=10 * size, **kwargs)

    def _gen_grid(self, height, width):
        # Create the grid
        self.grid = Grid(height, width)

        # Generate the surrounding walls
        self.horz_wall(0, 0)
        self.horz_wall(height - 1, 0)
        self.vert_wall(0, 0)
        self.vert_wall(0, width - 1)

        # Hallway walls

        left_wall = width // 2 - 2
        right_wall = width // 2 + 2

        self.vert_wall(0, left_wall, height=height)
        self.vert_wall(0, right_wall, height=height)

        self.rooms = []

        # Room splitting walls
        for n in range(0, 3):
            i = n * (height // 3)
            self.horz_wall(i, 0, width=left_wall)
            self.horz_wall(i, right_wall, width=width - right_wall)

            room_height = height // 3 + 1
            room_width = left_wall + 1
            self.rooms.append(_LockedRoom(
                (i, 0),
                (room_height, room_width),
                (i + 3, left_wall)
            ))
            self.rooms.append(_LockedRoom(
                (i, right_wall),
                (room_height, room_width),
                (i + 3, right_wall)
            ))

        # Choose one random room to be locked
        locked_room = self.rng.choice(self.rooms)
        locked_room.locked = True
        goal_i = self.rng.randint(locked_room.top[0] + 1, locked_room.top[0] + locked_room.size[0] - 1)
        goal_j = self.rng.randint(locked_room.top[1] + 1, locked_room.top[1] + locked_room.size[1] - 1)
        self[goal_i, goal_j] = Goal()

        # Assign the door colors
        colors = self.rng.choice(COLORS, size=len(self.rooms))
        for room, color in zip(self.rooms, colors):
            room.color = color
            if room.locked:
                self[room.door_pos] = Door(color, state='locked')
            else:
                self[room.door_pos] = Door(color)

        # Select a random room to contain the key
        while True:
            key_room = self.rng.choice(self.rooms)
            if key_room != locked_room:
                break
        key_pos = key_room.rand_pos(self)
        self[key_pos] = Key(locked_room.color)

        # Randomize the player start position and orientation
        self.place_agent(
            top=(0, left_wall),
            size=(height, right_wall - left_wall)
        )

        # Generate the mission string
        self.mission = (
            f'get the {locked_room.color} key from the {key_room.color} room, '
            f'unlock the {locked_room.color} door and go to the goal'
        )


class KeyCorridor(RoomGrid):
    """
    A ball is behind a locked door, the key is placed in a
    random room.

    This environment is similar to the locked room environment, but there are multiple registered environment configurations of increasing size, making it easier to use curriculum learning to train an agent to solve it. The agent has to pick up an object which is behind a locked door. The key is hidden in another room, and the agent has to explore the environment to find it. The mission string does not give the agent any clues as to where the key is placed.
    """
    _requires_language = False

    def __init__(
        self,
        num_rows=3,
        obj_type='ball',
        room_size=6,
        **kwargs
    ):
        self.obj_type = obj_type

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            max_steps=30 * room_size**2,
            **kwargs
        )

    def _gen_grid(self, height, width):
        super()._gen_grid(height, width)

        # Connect the middle column rooms into a hallway
        for i in range(1, self.num_rows):
            self.remove_wall(i, 1, 'up')

        # Add a locked door on the top left
        # Add an object behind the locked door
        room_idx = self.rng.randint(self.num_rows)
        door = self.add_door(room_idx, 2, door_idx='left', locked=True)
        self.obj = self.add_object(room_idx, 2, kind=self.obj_type)

        # Add a key in a random room on the left side
        self.add_object(self.rng.randint(self.num_rows), 0, 'key', door.color)

        # Place the agent in the middle
        self.place_agent(i=self.num_rows // 2, j=1)

        # Make sure all rooms are accessible
        self.connect_all()

        self.mission = f'pick up the {self.obj.color} {self.obj.type}'

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.actions[action] == 'pickup':
            if self.agent.is_carrying:
                if self.agent.carrying is self.obj:
                    reward = self._win_reward
                    done = True

        return obs, reward, done, info


class Unlock(RoomGrid):
    """
    The agent has to open a locked door.
    """
    _requires_language = False

    def __init__(self, **kwargs):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8 * room_size**2,
            **kwargs
        )

    def _gen_grid(self, height, width):
        super()._gen_grid(height, width)

        # Make sure the two rooms are directly connected by a locked door
        self.door = self.add_door(0, 0, door_idx='right', locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', self.door.color)

        self.place_agent(i=0, j=0)

        self.mission = 'open the door'

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.actions[action] == 'toggle':
            if self.door.is_open:
                reward = self._win_reward
                done = True

        return obs, reward, done, info


class UnlockPickup(RoomGrid):
    """
    The agent has to pick up a box which is placed in another room, behind a locked door.
    """
    _requires_language = False

    def __init__(self, **kwargs):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8 * room_size**2,
            **kwargs
        )

    def _gen_grid(self, height, width):
        super()._gen_grid(height, width)

        # Add a box to the room on the right
        self.obj = self.add_object(0, 1, kind='box')
        # Make sure the two rooms are directly connected by a locked door
        door = self.add_door(0, 0, door_idx='right', locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', door.color)

        self.place_agent(i=0, j=0)

        self.mission = f'pick up the {self.obj.color} {self.obj.type}'

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.actions[action] == 'pickup':
            if self.agent.is_carrying:
                if self.agent.carrying is self.obj:
                    reward = self._win_reward
                    done = True

        return obs, reward, done, info


class BlockedUnlockPickup(RoomGrid):
    """
    The agent has to pick up a box which is placed in another room, behind a locked door. The door is also blocked by a ball which the agent has to move before it can unlock the door. Hence, the agent has to learn to move the ball, pick up the key, open the door and pick up the object in the other room.
    """
    _requires_language = False

    def __init__(self, **kwargs):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=16 * room_size**2,
            **kwargs
        )

    def _gen_grid(self, height, width):
        super()._gen_grid(height, width)

        # Add a box to the room on the right
        self.obj = self.add_object(0, 1, kind='box')
        # Make sure the two rooms are directly connected by a locked door
        door = self.add_door(0, 0, door_idx='right', locked=True)
        # Block the door with a ball
        color = self.rng.choice(COLORS)
        self[door.pos[0], door.pos[1] - 1] = Ball(color)
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', door.color)
        self.place_agent(i=0, j=0)

        self.mission = f'pick up the {self.obj.color} {self.obj.type}'

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.actions[action] == 'pickup':
            if self.agent.is_carrying:
                if self.agent.carrying is self.obj:
                    reward = self._win_reward
                    done = True

        return obs, reward, done, info


class _ObstructedMaze(RoomGrid):
    """
    The agent has to pick up a box which is placed in a corner of a 3x3 maze. The doors are locked, the keys are hidden in boxes and doors are obstructed by balls.
    """
    _requires_language = False

    def __init__(self,
                 num_rows,
                 num_cols,
                 num_rooms_visited,
                 **kwargs
                 ):
        room_size = 6
        max_steps = 4 * num_rooms_visited * room_size**2

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            max_steps=max_steps,
            **kwargs
        )

    def _gen_grid(self, height, width):
        super()._gen_grid(height, width)

        # Define the color of the ball to pick up
        self.ball_to_find_color = COLORS[0]
        # Define the color of the balls that obstruct doors
        self.blocking_ball_color = COLORS[1]
        # Define the color of boxes in which keys are hidden
        self.box_color = COLORS[2]

        self.mission = f'pick up the {self.ball_to_find_color} ball'

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.actions[action] == 'pickup':
            if self.agent.is_carrying:
                if self.agent.carrying is self.obj:
                    reward = self._win_reward
                    done = True

        return obs, reward, done, info

    def add_door(self, i, j, door_idx='right', color=None, locked=False, key_in_box=False, blocked=False):
        """
        Add a door. If the door must be locked, it also adds the key.
        If the key must be hidden, it is put in a box. If the door must
        be obstructed, it adds a ball in front of the door.
        """

        door = super().add_door(i, j, door_idx=door_idx, color=color, locked=locked)

        if blocked:  # place a ball in front of the door
            if door_idx == 'right':
                offset = (0, -1)
            elif door_idx == 'down':
                offset = (-1, 0)
            elif door_idx == 'left':
                offset = (0, 1)
            elif door_idx == 'up':
                offset = (1, 0)
            pos = (door.pos[0] + offset[0], door.pos[1] + offset[1])
            self[pos] = Ball(self.blocking_ball_color)

        if locked:
            obj = Key(door.color)
            if key_in_box:
                box = Box(self.box_color) if key_in_box else None
                box.contains = obj
                obj = box
            self.place_in_room(i, j, obj)

        return door


class ObstructedMaze_1Dlhb(_ObstructedMaze):
    """
    A blue ball is hidden in a 2x1 maze. A locked door separates
    rooms. Doors are obstructed by a ball and keys are hidden in boxes.
    """
    _requires_language = False

    def __init__(self, key_in_box=True, blocked=True, **kwargs):
        self.key_in_box = key_in_box
        self.blocked = blocked

        super().__init__(
            num_rows=1,
            num_cols=2,
            num_rooms_visited=2,
            **kwargs
        )

    def _gen_grid(self, height, width):
        super()._gen_grid(height, width)

        self.add_door(0, 0, door_idx='right',
                      color=self.rng.choice(COLORS),
                      locked=True,
                      key_in_box=self.key_in_box,
                      blocked=self.blocked)

        self.obj = self.add_object(0, 1, 'ball', color=self.ball_to_find_color)
        self.place_agent(i=0, j=0)


class ObstructedMaze_Full(_ObstructedMaze):
    """
    A blue ball is hidden in one of the 4 corners of a 3x3 maze. Doors
    are locked, doors are obstructed by a ball and keys are hidden in
    boxes.
    """
    _requires_language = False

    def __init__(self, agent_room=(1, 1), key_in_box=True, blocked=True,
                 num_quarters=4, num_rooms_visited=25, **kwargs):
        self.agent_room = agent_room
        self.key_in_box = key_in_box
        self.blocked = blocked
        self.num_quarters = num_quarters

        super().__init__(
            num_rows=3,
            num_cols=3,
            num_rooms_visited=num_rooms_visited,
            **kwargs
        )

    def _gen_grid(self, height, width):
        super()._gen_grid(height, width)

        middle_room = (1, 1)
        # Define positions of "side rooms" i.e. rooms that are neither
        # corners nor the center.
        side_rooms = {
            'right': (1, 2),
            'down': (2, 1),
            'left': (1, 0),
            'up': (0, 1)
            }
        # Define all possible colors for doors
        door_colors = self.rng.choice(COLORS, size=len(side_rooms), replace=False)
        door_colors = {
            'right': door_colors[0],
            'down': door_colors[1],
            'left': door_colors[2],
            'up': door_colors[3]
            }

        side_rooms = list(side_rooms.items())[:self.num_quarters]
        for door_idx, side_room in side_rooms:
            # Add a door between the center room and the side room
            door_color = door_colors[door_idx]
            self.add_door(*middle_room, door_idx=door_idx, color=door_color, locked=False)

            for k in [-1, 1]:
                # Add a door to each side of the side room
                side_door_idx = self._door_idx(door_idx, k)
                side_door_color = door_colors[side_door_idx]
                self.add_door(*side_room, locked=True,
                              door_idx=side_door_idx,
                              color=side_door_color,
                              key_in_box=self.key_in_box,
                              blocked=self.blocked)

        corners = [(2, 0), (2, 2), (0, 2), (0, 0)]
        ball_room = corners[self.rng.randint(self.num_quarters)]

        self.obj = self.add_object(*ball_room, 'ball', color=self.ball_to_find_color)
        self.place_agent(i=self.agent_room[0], j=self.agent_room[1])


class DistShift(MiniGridEnv):
    """
    Distributional shift environment

    This environment is based on one of the DeepMind [AI safety gridworlds](https://github.com/deepmind/ai-safety-gridworlds). The agent starts in the top-left corner and must reach the goal which is in the top-right corner, but has to avoid stepping into lava on its way. The aim of this environment is to test an agent's ability to generalize. There are two slightly different variants of the environment, so that the agent can be trained on one variant and tested on the other.
    """

    def __init__(
        self,
        width=9,
        height=7,
        agent_start_pos=(1,1),
        agent_start_state='right',
        strip2_row=2,
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_state = agent_start_state
        self.goal_pos = (1, width - 2)
        self.strip2_row = strip2_row

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            **kwargs
        )

    def _gen_grid(self, height, width):
        # Create an empty grid
        self.grid = Grid(height, width)

        # Generate the surrounding walls
        self.wall_rect(0, 0, height, width)

        # Place a goal square in the bottom-right corner
        self[self.goal_pos] = Goal()

        # Place the lava rows
        for j in range(self.width - 6):
            self[1, 3 + j] = Lava()
            self[self.strip2_row, 3 + j] = Lava()

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent.pos = self.agent_start_pos
            self.agent.state = self.agent_start_state
        else:
            self.place_agent()

        self.mission = 'get to the green goal square'


class LavaGap(MiniGridEnv):
    """
    The agent has to reach the green goal square at the opposite corner of the room, and must pass through a narrow gap in a vertical strip of deadly lava. Touching the lava terminate the episode with a zero reward. This environment is useful for studying safety and safe exploration.
    """

    def __init__(self, size=7, obstacle_type=Lava, **kwargs):
        self.obstacle_type = obstacle_type
        super().__init__(
            height=size,
            width=size,
            max_steps=4 * size**2,
            **kwargs
        )

    def _gen_grid(self, height, width):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(height, width)

        # Generate the surrounding walls
        self.wall_rect(0, 0, height, width)

        # Place the agent in the top-left corner
        self.agent.pos = (1, 1)
        self.agent.state = 'right'

        # Place a goal square in the bottom-right corner
        self[height - 2, width - 2] = Goal()

        # Generate and store random gap position
        gap_pos = (self.rng.randint(1, height - 1),
                   self.rng.randint(2, width - 2))

        # Place the obstacle wall
        self.vert_wall(1, gap_pos[1], height=height - 2,
                       obj=self.obstacle_type)

        # Put a hole in the wall
        self[gap_pos].clear()

        if type(self.obstacle_type) is Lava:
            self.mission = 'avoid the lava and get to the green goal square'
        else:
            self.mission = 'find the opening and get to the green goal square'


class _Crossing(MiniGridEnv):
    """
    Environment with wall or lava obstacles, sparse reward.
    """

    def __init__(self, size=9, num_crossings=1, obstacle_type=Lava, **kwargs):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        super().__init__(
            height=size,
            width=size,
            max_steps=4 * size * size,
            **kwargs
        )

    def _gen_grid(self, height, width):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(height, width)

        # Generate the surrounding walls
        self.wall_rect(0, 0, height, width)

        # Place the agent in the top-left corner
        self.agent.pos = (1, 1)
        self.agent.state = 'right'

        # Place a goal square in the bottom-right corner
        self[height - 2, width - 2] = Goal()

        # Place obstacles (lava or walls)

        # Lava rivers or walls specified by direction and position in grid
        rivers = [('v', i) for i in range(2, width - 2, 2)]
        rivers += [('h', j) for j in range(2, height - 2, 2)]
        self.rng.shuffle(rivers)
        rivers = rivers[:self.num_crossings]  # sample random rivers
        rivers_v = sorted([pos for d, pos in rivers if d == 'v'])
        rivers_h = sorted([pos for d, pos in rivers if d == 'h'])
        obstacle_pos = itertools.chain(
            itertools.product(range(1, height - 1), rivers_v),
            itertools.product(rivers_h, range(1, width - 1)),
        )
        for pos in obstacle_pos:
            self[pos] = self.obstacle_type()

        # Sample path to goal
        path = ['v'] * len(rivers_v) + ['h'] * len(rivers_h)
        self.rng.shuffle(path)

        # Create openings
        limits_h = [0] + rivers_h + [height - 1]
        limits_v = [0] + rivers_v + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction == 'h':
                i = limits_h[room_i + 1]
                j = self.rng.choice(
                    range(limits_v[room_j] + 1, limits_v[room_j + 1]))
                room_i += 1
            elif direction == 'v':
                i = self.rng.choice(
                    range(limits_h[room_i] + 1, limits_h[room_i + 1]))
                j = limits_v[room_j + 1]
                room_j += 1
            self[i, j].clear()

        if type(self.obstacle_type) is Lava:
            self.mission = 'avoid the lava and get to the green goal square'
        else:
            self.mission = 'find the opening and get to the green goal square'


class LavaCrossing(_Crossing):
    """
    The agent has to reach the green goal square on the other corner of the room while avoiding rivers of deadly lava which terminate the episode in failure. Each lava stream runs across the room either horizontally or vertically, and has a single crossing point which can be safely used; Luckily, a path to the goal is guaranteed to exist. This environment is useful for studying safety and safe exploration.
    """

    def __init__(self, size=9, num_crossings=1, **kwargs):
        super().__init__(size=size, num_crossings=num_crossings, obstacle_type=Lava, **kwargs)


class SimpleCrossing(_Crossing):
    """
    Similar to the LavaCrossing environment, the agent has to reach the green goal square on the other corner of the room, however lava is replaced by walls. This MDP is therefore much easier and and maybe useful for quickly testing your algorithms.
    """

    def __init__(self, size=11, num_crossings=5, **kwargs):
        super().__init__(size=size, num_crossings=num_crossings,
                         obstacle_type=Wall, **kwargs)


class DynamicObstacles(MiniGridEnv):
    """
    This environment is an empty room with moving obstacles. The goal of the agent is to reach the green goal square without colliding with any obstacle. A large penalty is subtracted if the agent collides with an obstacle and the episode finishes. This environment is useful to test Dynamic Obstacle Avoidance for mobile robots with Reinforcement Learning in Partial Observability.
    """

    def __init__(
            self,
            size=8,
            agent_start_pos=(1, 1),
            agent_start_state='right',
            n_obstacles=4,
            **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_state = agent_start_state

        # Reduce obstacles if there are too many
        if n_obstacles <= size / 2 + 1:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size / 2)
        super().__init__(
            height=size,
            width=size,
            max_steps=4 * size * size,
            **kwargs
        )
        # Allow only 3 actions permitted: left, right, forward
        self.action_space = gym.spaces.Discrete(3)

    def _gen_grid(self, height, width):
        # Create an empty grid
        self.grid = Grid(height, width)

        # Generate the surrounding walls
        self.wall_rect(0, 0, height, width)

        # Place a goal square in the bottom-right corner
        self[height - 2, width - 2] = Goal()

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent.pos = self.agent_start_pos
            self.agent.state = self.agent_start_state
        else:
            self.place_agent()

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], max_tries=100)

        self.mission = 'get to the green goal square'

    def step(self, action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # Check if there is an obstacle in front of the agent
        front_cell = self[self.agent.front_pos]
        not_clear = front_cell.entity is not None and front_cell.entity.type != 'goal'

        obs, reward, done, info = super().step(action)

        # If the agent tries to walk over an obstacle
        if self.actions[action] == 'forward' and not_clear:
            reward = self._lose_reward
            done = True
            return obs, reward, done, info

        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].pos
            top = tuple(map(operator.add, old_pos, (-1, -1)))

            try:
                self.place_obj(self.obstacles[i_obst], top=top, size=(3,3), max_tries=100)
                self[old_pos].clear()
            except RecursionError:
                pass

        return obs, reward, done, info


class Playground(MiniGridEnv):
    """
    Environment with multiple rooms and random objects.
    This environment has no specific goals or rewards.
    """

    def __init__(self, size=19, max_steps=100, **kwargs):
        super().__init__(height=size, width=size, max_steps=max_steps, **kwargs)

    def _gen_grid(self, height, width):
        # Create the grid
        self.grid = Grid(height, width)

        # Generate the surrounding walls
        self.horz_wall(0, 0)
        self.horz_wall(height - 1, 0)
        self.vert_wall(0, 0)
        self.vert_wall(0, width - 1)

        room_width = width // 3
        room_height = height // 3

        # For each row of rooms
        for i in range(3):
            # For each column
            for j in range(3):

                i_top = i * room_height
                j_left = j * room_width
                i_bottom = i_top + room_height
                j_right = j_left + room_width

                # Right wall and door
                if j + 1 < 3:
                    self.vert_wall(i_top, j_right, height=room_height)
                    pos = (self.rng.randint(i_top + 1, i_bottom - 1), j_right)
                    color = self.rng.choice(COLORS)
                    self[pos] = Door(color)

                # Bottom wall and door
                if i + 1 < 3:
                    self.horz_wall(i_bottom, j_left, width=room_width)
                    pos = (i_bottom, self.rng.randint(j_left + 1, j_right - 1))
                    color = self.rng.choice(COLORS)
                    self[pos] = Door(color)

        # Place random objects in the world
        types = ['key', 'ball', 'box']
        for i in range(0, 12):
            obj_type = self.rng.choice(types)
            obj_color = self.rng.choice(COLORS)
            if obj_type == 'key':
                obj = Key(obj_color)
            elif obj_type == 'ball':
                obj = Ball(obj_color)
            elif obj_type == 'box':
                obj = Box(obj_color)
            self.place_obj(obj)

        # Randomize the player start position and orientation
        self.place_agent()

        # No explicit mission in this environment
        self.mission = ''


class RandomObjects(MiniGridEnv):
    """
    This environment is a blank grid filled with randomly placed objects (including wall elements). Useful for curriculum learning as the first learning stage.
    """

    def __init__(self, size=16, max_steps=1000, **kwargs):
        super().__init__(height=size, width=size, max_steps=max_steps, **kwargs)

    def _gen_grid(self, height, width):
        # Create an empty grid
        self.grid = Grid(height, width)

        # Place a goal square at a random location
        self.place_obj(Goal())

        # Place random objects in the world
        mean_n_objs = int(height * width * .4)
        n_objs = int(self.rng.normal(mean_n_objs, mean_n_objs // 2))
        n_objs = np.clip(n_objs, mean_n_objs // 5, mean_n_objs * 9 // 5)
        for i in range(n_objs):
            obj = self.make_obj()
            self.place_obj(obj)

        # Randomize the player start position and orientation
        agent_pos = (height // 2, width // 2)
        self[agent_pos].clear()
        self.agent.pos = agent_pos
        self.agent.state = self.rng.choice(self.agent.STATES)

        self.mission = 'get to a green goal square'

    def make_obj(self):
        type_ = self.rng.choice(OBJECTS)
        color = self.rng.choice(COLORS)

        if type_ in ['wall', 'goal', 'lava']:
            obj = entities.make(type_)
        elif type_ == 'door':
            state = self.rng.choice(entities.Door.STATES)
            obj = entities.make(type_, color=color, state=state)
        elif type_ == 'box':
            if self.rng.random() < .5:
                contains = None
            else:
                contains = self.make_obj()
            obj = entities.make(type_, color=color, contains=contains)
        else:
            obj = entities.make(type_, color=color)

        return obj


# Register all environments with OpenAI gym
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__ and not name.startswith('_'):

        gym.envs.registration.register(
            id=f'MiniGrid-{name}-v1',
            entry_point=f'gym_minigrid.envs:{name}',
            reward_threshold=.95,
        )
