import numpy as np

# Object types
OBJECTS = ['wall', 'door', 'key', 'ball', 'box', 'goal', 'lava']
# Allowed object colors
COLORS = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'grey']


class Agent(object):

    STATES = ['right', 'down', 'left', 'up']
    ACTIONS = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
    DIR_TO_VEC = {
        'right': (0, 1),
        'down': (1, 0),
        'left': (0, -1),
        'up': (-1, 0),
    }

    def __init__(self, view_size=7, state='right'):
        self.type = 'agent'
        self.color = 'red'
        self.state = state
        # Number of cells (width and height) in the agent view
        self.view_size = view_size
        self.visited = set()

    def reset(self):
        # Item picked up, being carried
        self.carrying = None
        # Current position of the agent
        self._pos = None

    def __str__(self):
        r = super().__str__()
        return f'{r}, carrying: {self.carrying}'

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    @property
    def is_carrying(self):
        return self.carrying is not None

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, pos):
        self._pos = pos
        self.visited.add(pos)

    @property
    def dir(self):
        return self.STATES.index(self.state)

    @property
    def idir(self):
        return self.DIR_TO_VEC[self.state][0]

    @property
    def jdir(self):
        return self.DIR_TO_VEC[self.state][1]

    def rotate_left(self):
        idx = self.STATES.index(self.state)
        self.state = self.STATES[(idx - 1) % len(self.STATES)]

    def rotate_right(self):
        idx = self.STATES.index(self.state)
        self.state = self.STATES[(idx + 1) % len(self.STATES)]

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """
        if self.state == 'right':
            offset = (0, 1)
        elif self.state == 'down':
            offset = (1, 0)
        elif self.state == 'left':
            offset = (0, -1)
        elif self.state == 'up':
            offset = (-1, 0)
        return (self.pos[0] + offset[0], self.pos[1] + offset[1])

    @property
    def view_box(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        if self.state == 'right':
            top_i = self.pos[0] - self.view_size // 2
            top_j = self.pos[1]
        elif self.state == 'down':
            top_i = self.pos[0]
            top_j = self.pos[1] - self.view_size // 2
        elif self.state == 'left':
            top_i = self.pos[0] - self.view_size // 2
            top_j = self.pos[1] - self.view_size + 1
        elif self.state == 'up':
            top_i = self.pos[0] - self.view_size + 1
            top_j = self.pos[1] - self.view_size // 2

        bottom_i = top_i + self.view_size
        bottom_j = top_j + self.view_size

        return slice(top_i, bottom_i), slice(top_j, bottom_j)


class WorldObj(object):
    """
    Base class for grid world objects
    """

    def __init__(self, type_, color, state=None):
        self.type = type_
        self.color = color
        self.state = state
        self.contains = None
        # self.pos = None

    def encode(self):
        return [self.type, self.color, self.state]

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False


class Wall(WorldObj):

    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False


class Door(WorldObj):

    STATES = ['open', 'closed', 'locked']

    def __init__(self, color='blue', state='closed'):
        super().__init__('door', color)
        self.state = state

    @property
    def is_open(self):
        return self.state == 'open'

    @is_open.setter
    def is_open(self, open_):
        self.state = 'open' if open_ else 'closed'

    @property
    def is_locked(self):
        return self.state == 'locked'

    @is_locked.setter
    def is_locked(self, locked):
        self.state = 'locked' if locked else 'closed'

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if env.agent.is_carrying:
                if env.agent.carrying.type == 'key' and env.agent.carrying.color == self.color:
                    self.is_locked = False
                    self.is_open = True
                    return True
            return False

        self.is_open = not self.is_open
        return True


class Key(WorldObj):

    def __init__(self, color='blue'):
        super().__init__('key', color)

    def can_pickup(self):
        return True


class Ball(WorldObj):

    def __init__(self, color='blue'):
        super().__init__('ball', color)

    def can_pickup(self):
        return True


class Box(WorldObj):

    def __init__(self, color='blue', contains=None):
        super().__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def toggle(self, env, pos):
        # Replace the box by its contents
        if self.contains is None:
            env[pos].clear()
        else:
            env[pos] = self.contains
        return True


class Goal(WorldObj):

    def __init__(self, color='green'):
        super().__init__('goal', 'green')

    def can_overlap(self):
        return True


class Lava(WorldObj):

    def __init__(self, color='red'):
        super().__init__('lava', 'red')

    def can_overlap(self):
        return True


def make(type_, *args, **kwargs):
    """
    Returns an intialized object given its type (string name).
    """
    if type_ not in OBJECTS:
        raise ValueError(f'Object type must be one of: {OBJECTS}, but got {type_}')
    obj = globals()[type_.capitalize()]
    return obj(*args, **kwargs)
