from gym_minigrid.minigrid import Cell, Grid, MiniGridEnv
from gym_minigrid.entities import Ball, Box, Door, Key, COLORS, make


def _reject_next_to(env, pos):
    """
    Function to filter out object positions that are right next to
    the agent's starting point
    """

    si, sj = env.agent.pos
    i, j = pos
    d = abs(si - i) + abs(sj - j)
    return d < 2


class Room(object):

    def __init__(self, top, size):
        # Top-left corner and size (tuples)
        self.top = top
        self.size = size

        # List of door objects and door positions
        # Order of the doors is right, down, left, up
        self.doors = {'right': None, 'down': None, 'left': None, 'up': None}
        self.door_pos = {'right': None, 'down': None, 'left': None, 'up': None}

        # List of rooms adjacent to this one
        # Order of the neighbors is right, down, left, up
        self.neighbors = {'right': None, 'down': None, 'left': None, 'up': None}

        # Indicates if this room is behind a locked door
        self.locked = False

        # List of objects contained
        self.objs = []


class RoomGrid(MiniGridEnv):
    """
    Environment with multiple rooms and random objects.
    This is meant to serve as a base class for other environments.
    """
    ORIENTATIONS = ['right', 'down', 'left', 'up']

    def __init__(
        self,
        room_size=7,
        num_rows=3,
        num_cols=3,
        max_steps=100,
        seed=0
    ):
        assert room_size > 0
        assert room_size >= 3
        assert num_rows > 0
        assert num_cols > 0
        self.room_size = room_size
        self.num_rows = num_rows
        self.num_cols = num_cols

        height = (room_size - 1) * num_rows + 1
        width = (room_size - 1) * num_cols + 1

        # By default, this environment has no mission
        self.mission = ''

        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,
            seed=seed
        )

    def room_from_pos(self, i, j):
        """Get the room a given position maps to"""

        assert i >= 0
        assert j >= 0

        i //= (self.room_size - 1)
        j //= (self.room_size - 1)

        assert i < self.num_rows
        assert j < self.num_cols

        return self.room_grid[i][j]

    def _gen_grid(self, height, width):
        # Create the grid
        self.grid = Grid(height, width)

        self.room_grid = []

        # For each row of rooms
        for i in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for j in range(0, self.num_cols):
                room = Room(
                    (i * (self.room_size - 1), j * (self.room_size - 1)),
                    (self.room_size, self.room_size)
                )
                row.append(room)

                # Generate the walls for this room
                self.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for i in range(0, self.num_rows):
            # For each column of rooms
            for j in range(0, self.num_cols):
                room = self.room_grid[i][j]

                i_l, j_l = (room.top[0] + 1, room.top[1] + 1)
                i_m, j_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions
                if j < self.num_cols - 1:
                    room.neighbors['right'] = self.room_grid[i][j + 1]
                    room.door_pos['right'] = (self.rng.randint(i_l, i_m), j_m)
                if i < self.num_rows - 1:
                    room.neighbors['down'] = self.room_grid[i + 1][j]
                    room.door_pos['down'] = (i_m, self.rng.randint(j_l, j_m))
                if j > 0:
                    room.neighbors['left'] = self.room_grid[i][j - 1]
                    room.door_pos['left'] = room.neighbors['left'].door_pos['right']
                if i > 0:
                    room.neighbors['up'] = self.room_grid[i - 1][j]
                    room.door_pos['up'] = room.neighbors['up'].door_pos['down']

        # The agent starts in the middle, facing right
        self.agent.pos = (
            (self.num_rows // 2) * (self.room_size - 1) + (self.room_size // 2),
            (self.num_cols // 2) * (self.room_size - 1) + (self.room_size // 2)
        )
        self.agent.state = 'right'

    def place_in_room(self, i, j, obj):
        """
        Add an existing object to room (i, j)
        """
        room = self.room_grid[i][j]
        self.place_obj(
            obj,
            room.top,
            room.size,
            reject_fn=_reject_next_to,
            max_tries=1000
        )
        room.objs.append(obj)

    def add_object(self, i, j, kind=None, color=None):
        """
        Add a new object to room (i, j)
        """
        if kind is None:
            kind = self.rng.choice(['key', 'ball', 'box'])
        if color is None:
            color = self.rng.choice(COLORS)
        obj = make(kind, color)
        self.place_in_room(i, j, obj)
        return obj

    def add_door(self, i, j, door_idx=None, color=None, locked=None):
        """
        Add a door to a room, connecting it to a neighbor
        """

        room = self.room_grid[i][j]

        if door_idx is None:
            # Need to make sure that there is a neighbor along this wall
            # and that there is not already a door
            while True:
                door_idx = self.rng.choice(self.ORIENTATIONS)
                if room.neighbors[door_idx] and room.doors[door_idx] is None:
                    break

        if room.doors[door_idx] is not None:
            raise IndexError(f'door {door_idx} already exists')

        if color is None:
            color = self.rng.choice(COLORS)

        if locked is None:
            locked = self.rng.rand() > .5

        room.locked = locked
        door = Door(color, state='locked' if locked else 'closed')

        pos = room.door_pos[door_idx]
        self[pos] = door

        room.doors[door_idx] = door
        room.neighbors[door_idx].doors[self._door_idx(door_idx, 2)] = door

        return door

    def _door_idx(self, door_idx, offset):
        idx = self.ORIENTATIONS.index(door_idx)
        door_idx = self.ORIENTATIONS[(idx + offset) % len(self.ORIENTATIONS)]
        return door_idx

    def remove_wall(self, i, j, wall_idx):
        """
        Remove a wall between two rooms
        """
        room = self.room_grid[i][j]

        if room.doors[wall_idx] is not None:
            raise ValueError('door exists on this wall')
        if not room.neighbors[wall_idx]:
            raise ValueError(f'invalid wall: {wall_idx}')

        neighbor = room.neighbors[wall_idx]

        ti, tj = room.top
        w, h = room.size

        # Ordering of walls is right, down, left, up
        if wall_idx == 'right':
            for i in range(1, h - 1):
                self[ti + i, tj + w - 1].clear()
        elif wall_idx == 'down':
            for j in range(1, w - 1):
                self[ti + h - 1, tj + j].clear()
        elif wall_idx == 'left':
            for i in range(1, h - 1):
                self[ti + i, tj].clear()
        elif wall_idx == 'up':
            for j in range(1, w - 1):
                self[ti, tj + j].clear()
        else:
            raise ValueError(f'invalid wall: {wall_idx}')

        # Mark the rooms as connected
        room.doors[wall_idx] = True
        neighbor.doors[self._door_idx(wall_idx, 2)] = True

    def place_agent(self, i=None, j=None, rand_dir=True):
        """
        Place the agent in a room
        """

        if i is None:
            i = self.rng.randint(self.num_rows)
        if j is None:
            j = self.rng.randint(self.num_cols)

        room = self.room_grid[i][j]

        # Find a position that is not right in front of an object
        while True:
            super().place_agent(top=room.top, size=room.size, rand_dir=rand_dir, max_tries=1000)
            front_cell = self[self.agent.front_pos]
            if front_cell.entity is None or front_cell.entity.type == 'wall':
                break
            else:
                self[self.agent.pos].clear()

        return self.agent.pos

    def connect_all(self, door_colors=COLORS, max_itrs=5000):
        """
        Make sure that all rooms are reachable by the agent from its
        starting position
        """
        start_room = self.room_from_pos(*self.agent.pos)

        added_doors = []

        def find_reach():
            reach = set()
            stack = [start_room]
            while len(stack) > 0:
                room = stack.pop()
                if room in reach:
                    continue
                reach.add(room)
                for ori in self.ORIENTATIONS:
                    if room.doors[ori]:
                        stack.append(room.neighbors[ori])
            return reach

        num_itrs = 0

        while True:
            # This is to handle rare situations where random sampling produces
            # a level that cannot be connected, producing in an infinite loop
            if num_itrs > max_itrs:
                raise RecursionError('connect_all failed')
            num_itrs += 1

            # If all rooms are reachable, stop
            reach = find_reach()
            if len(reach) == self.num_rows * self.num_cols:
                break

            # Pick a random room and door position
            i = self.rng.randint(0, self.num_rows)
            j = self.rng.randint(0, self.num_cols)
            k = self.rng.choice(self.ORIENTATIONS)
            room = self.room_grid[i][j]

            # If there is already a door there, skip
            if not room.door_pos[k] or room.doors[k]:
                continue

            if room.locked or room.neighbors[k].locked:
                continue

            color = self.rng.choice(door_colors)
            door = self.add_door(i, j, k, color, False)
            added_doors.append(door)

        return added_doors

    def add_distractors(self, i=None, j=None, num_distractors=10, all_unique=True):
        """
        Add random objects that can potentially distract/confuse the agent.
        """
        raise NotImplementedError
        # Collect a list of existing objects
        objs = []
        for row in self.room_grid:
            for room in row:
                for obj in room.objs:
                    objs.append((obj.type, obj.color))

        # List of distractors added
        dists = []

        while len(dists) < num_distractors:
            color = self.rng.choice(COLORS)
            type = self.rng.choice(['key', 'ball', 'box'])
            obj = (type, color)

            if all_unique and obj in objs:
                continue

            # Add the object to a random room if no room specified
            room_i = i
            room_j = j
            if room_i is None:
                room_i = self.rng.randint(0, self.num_rows)
            if room_j is None:
                room_j = self.rng.randint(0, self.num_cols)

            dist, pos = self.add_object(room_i, room_j, *obj)

            objs.append(obj)
            dists.append(dist)

        return dists
