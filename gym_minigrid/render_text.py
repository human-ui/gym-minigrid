import curses
import sty

import numpy as np


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


class ANSI(ASCII):

    # Map of object types to unicode symbols
    OBJECT_TO_STR = {
        'empty': '\u2588',
        'wall': '\u2588',
        'door': {
            'open': '\u2337',
            'closed': '\u2338',
            'locked': '\u236F'
        },
        'key': '\u2642',
        'ball': '\u25CF',
        'box': '\u25EB',
        'goal': '\u2588',  # \u26DD',  # '\u2612',
        'lava': '\u2591',
        'agent':
            {
            'right': '\u25B6',
            'down': '\u25BC',
            'left': '\u25C0',
            'up': '\u25B2'
        }
    }

    def __init__(self, env):
        super().__init__(env)

    def _color(self, color, kind='fg', bright=False):
        if color == 'grey':
            color = 'white'

        styling = getattr(sty, kind)
        if bright:
            color = styling('li_' + color)
        else:
            color = styling(color)

        return color

    def __call__(self):
        output = ''
        mask = self.env.visible()
        for pos, cell in np.ndenumerate(self.env.grid):
            bright = mask[pos]

            bg_color = self._color(cell.color, kind='bg', bright=bright)

            if pos == self.env.agent.pos:
                entity = self.env.agent
            else:
                entity = cell.entity

            if entity is not None:
                fg_color = self._color(entity.color, kind='fg', bright=bright)
                obj = self.OBJECT_TO_STR[entity.type]
                if isinstance(obj, dict):
                    obj = obj[entity.state]
            else:
                fg_color = self._color('black', kind='fg', bright=bright)
                obj = self.OBJECT_TO_STR['empty']

            output += bg_color + fg_color + obj + sty.rs.all
            if pos[1] == self.env.width - 1:
                output += '\n'

        return output


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
