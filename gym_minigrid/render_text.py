import curses, functools
import sty

import numpy as np

import gym_minigrid.encoding
CH = gym_minigrid.encoding.Channels()

ATTRS = {k:v for a in gym_minigrid.encoding.ATTRS if isinstance(a, dict) for k,v in a.items()}


class ASCII(object):

    OBJECT_TO_STR = {
        'empty': ' ',
        'wall': 'W',
        'door': {
                'open': '_',
                'closed': 'D',
                'locked': 'L'
        },
        'key': 'K',
        'ball': 'A',
        'box': 'B',
        'goal': 'G',
        'lava': 'V',
    }

    AGENT_DIR_TO_STR = {
        'right': '>',
        'down': 'V',
        'left': '<',
        'up': '^'
    }

    def __init__(self, env):
        self.env = env

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """
        output = ''

        for j in range(self.env.grid.height):

            for i in range(self.env.grid.width):
                if i == self.env.agent_pos[0] and j == self.env.agent_pos[1]:
                    output += 2 * self.AGENT_DIR_TO_STR[self.env.agent_dir]
                    continue

                c = self.grid.get(i, j)

                if c is None:
                    output += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        output += '__'
                    elif c.is_locked:
                        output += 'L' + c.color[0].upper()
                    else:
                        output += 'D' + c.color[0].upper()
                    continue

                output += self.OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.env.grid.height - 1:
                output += '\n'

        return output


class Render(object):

    def __init__(self, grid):
        self.grid = grid

    def get_attr(self, i, j, attr):
        attrs = getattr(CH, attr)
        idx = np.argmax(self.grid[attrs, i, j])
        return ATTRS[attr][idx]

    def convert(self):
        canvas = np.empty((self.grid.shape[1], self.grid.shape[2]), dtype='<U20')

        for i in range(self.grid.shape[1]):
            for j in range(self.grid.shape[2]):
                channels = self.grid[:, i, j]
                highlight = channels[CH.visible]

                if channels[CH.agent_pos]:
                    state = self.get_attr(i, j, 'agent_state')
                    canvas[i][j] = self.agent('red', state=state, highlight=highlight)

                else:
                    if channels[CH.empty] < .5:
                        obj_type = self.get_attr(i, j, 'object_type')
                        obj_color = self.get_attr(i, j, 'object_color')
                        door_state = self.get_attr(i, j, 'door_state')

                        f = getattr(self, obj_type)
                        if obj_type != 'door':
                            canvas[i][j] = f(obj_color, highlight=highlight)
                        else:
                            canvas[i][j] = f(obj_color, state=door_state, highlight=highlight)

                    else:
                        canvas[i][j] = self.empty('black', highlight=highlight)
        return canvas


class ANSI(Render):

    def setup(func):
        @functools.wraps(func)
        def wrap(self, color, highlight=False, **kwargs):
            bg_color = self._color('black', kind='bg', highlight=highlight)
            fg_color = self._color(color, kind='fg', highlight=highlight)
            obj = func(self, color, **kwargs)
            return bg_color + fg_color + obj + sty.rs.all
        return wrap

    def _color(self, color, kind='fg', highlight=False):
        if color == 'grey':
            color = 'white'

        styling = getattr(sty, kind)
        if highlight:
            color = styling('li_' + color)
        else:
            color = styling(color)

        return color

    def __str__(self):
        canvas = self.convert()
        output = '\n'.join(''.join(item) for item in canvas)
        return output

    @setup
    def agent(self, color, state, highlight=False):
        agent = {
            'right': '\u25B6',
            'down': '\u25BC',
            'left': '\u25C0',
            'up': '\u25B2'
        }
        return agent[state]

    @setup
    def empty(self, color, highlight=False):
        return '\u2588'

    @setup
    def wall(self, color, highlight=False):
        return '\u2588'

    @setup
    def door(self, color, state, highlight=False):
        door = {
            'open': '\u2337',
            'closed': '\u2338',
            'locked': '\u236F'
        }
        return door[state]

    @setup
    def key(self, color, highlight=False):
        return '\u2642'

    @setup
    def ball(self, color, highlight=False):
        return '\u25CF'

    @setup
    def box(self, color, highlight=False):
        return '\u25EB'

    @setup
    def goal(self, color, highlight=False):
        return '\u2588'

    @setup
    def lava(self, color, highlight=False):
        return '\u2591'


class Curses4bit(ANSI):

    def init_colors(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        """
        curses.start_color()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_BLACK)
        colors = {'black': curses.color_pair(1)}
        for i, color_name in enumerate(self.colors):
            if color_name == 'grey':
                color = 'white'
            else:
                color = color_name
            fc = getattr(curses, f'COLOR_{color.upper()}')
            curses.init_pair(2 * i + 3, fc, curses.COLOR_BLACK)
            colors[color_name] = curses.color_pair(2 * i + 3)

        self.colors = colors

    def __call__(self, screen, pos=(0, 0)):
        to_highlight = self.env.vis_mask_abs()

        for j in range(self.env.grid.height):
            for i in range(self.env.grid.width):

                if i == self.env.agent_pos[0] and j == self.env.agent_pos[1]:
                    rep = self.AGENT_DIR_TO_STR[self.env.agent_dir]
                    color = 'red'
                else:
                    c = self.grid.get(i, j)

                    if c is None:
                        rep = self.OBJECT_TO_STR['floor']
                        color = 'black'
                    else:
                        rep = self.OBJECT_TO_STR[c.type]
                        color = c.color
                        if c.type == 'door':
                            if c.is_open:
                                rep = rep['open']
                            elif c.is_locked:
                                rep = rep['locked']
                            else:
                                rep = rep['closed']

                if (i, j) in to_highlight:
                    color = self.colors_highlight[color]
                else:
                    color = self.colors[color]
                self.screen.addstr(j + pos[1], i + pos[0], rep, color)


class Curses8bit(ANSI):
    """
    Note: properly renders colors only in some terminals
    E.g., works fine in iTerm2 but not in xterm or hyper
    """

    def init_colors(self):
        # define background color
        curses.start_color()
        curses.init_color(0, 300, 300, 300)  # background black-ish
        # curses.init_color(1, 1000, 1000, 1000)  # white
        curses.init_pair(1, 0, 0)

        # define highlight colors
        h = .75
        high_bg = int(1000 * (1 - h))

        curses.init_color(2, 0, 0, 0)  # black
        curses.init_pair(2, 2, 2)
        curses.init_color(3, high_bg, high_bg, high_bg)  # highlighted gray
        curses.init_pair(3, 3, 3)

        # define object colors
        colors = {'black': curses.color_pair(2)}
        colors_highlight = {'black': curses.color_pair(3)}
        for idx, (color, (r, g, b)) in enumerate(self.colors.items()):
            cidx = 2 * idx + 4
            curses.init_color(cidx,
                              int(r / 255 * 1000),
                              int(g / 255 * 1000),
                              int(b / 255 * 1000))
            curses.init_pair(cidx, cidx, 2)
            colors[color] = curses.color_pair(cidx)

            curses.init_color(cidx + 1,
                              int(r / 255 * 1000 * h + high_bg),
                              int(g / 255 * 1000 * h + high_bg),
                              int(b / 255 * 1000 * h + high_bg))
            curses.init_pair(cidx + 1, cidx + 1, 3)
            colors_highlight[color] = curses.color_pair(cidx + 1)

        self.colors = colors
        self.colors_highlight = colors_highlight

    def __call__(self, screen, pos=(0, 0)):
        screen.bkgd(' ', curses.color_pair(1))
        super().__call__(screen, pos)
