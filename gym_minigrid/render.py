import curses, functools
import sty

import numpy as np

import matplotlib.colors as colors
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt

from gym_minigrid import window, encoding
CH = encoding.Channels()


class Render(object):

    def render(self, array):

        def get_attr(i, j, attr):
            attrs = getattr(CH, attr)
            idx = np.argmax(array[attrs, i, j])
            return CH.attrs[attr][idx]

        canvas = np.empty((array.shape[1], array.shape[2]), dtype=np.object)

        for i in range(array.shape[1]):
            if i > -1:
                for j in range(array.shape[2]):
                    channels = array[:, i, j]
                    highlight = channels[CH.visible]

                    if channels[CH.agent_pos]:
                        state = get_attr(i, j, 'agent_state')
                        canvas[i][j] = self.agent('red', state=state, highlight=highlight)

                    else:
                        if channels[CH.empty] < .5:
                            obj_type = get_attr(i, j, 'object_type')
                            obj_color = get_attr(i, j, 'object_color')
                            door_state = get_attr(i, j, 'door_state')
                            # if obj_type != 'wall':
                            #     breakpoint()

                            f = getattr(self, obj_type)
                            if obj_type != 'door':
                                canvas[i][j] = f(obj_color, highlight=highlight)
                            else:
                                canvas[i][j] = f(obj_color, state=door_state, highlight=highlight)

                        else:
                            canvas[i][j] = self.empty('black', highlight=highlight)

        return canvas


class ANSI(Render):

    def render(self, array):
        output = super().render(array)
        output = '\n'.join(''.join(item) for item in output)
        return output

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

    @setup
    def agent(self, color, state, highlight=False):
        agent = {
            'right': '\u2192',
            'down': '\u2193',
            'left': '\u2190',
            'up': '\u2191'
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
        return '\u21AC'  # \u2642'

    @setup
    def ball(self, color, highlight=False):
        return '\u25CF'

    @setup
    def box(self, color, highlight=False):
        return '\u25A0'

    @setup
    def goal(self, color, highlight=False):
        return '\u2B51'

    @setup
    def lava(self, color, highlight=False):
        return '\u2591'


class Matplotlib(Render):

    def render(self, array, ax=None):
        canvas = super().render(array)
        if ax is None:
            _, ax = plt.subplots()

        width = 32 * canvas.shape[1]
        height = 32 * canvas.shape[0]
        pc = [Rectangle((0, 0), width, height, color='black')]

        for i in range(1, canvas.shape[0]):
            pc.append(Polygon([(32 * i, 0), (32 * i, width)], color='grey'))
        for j in range(1, canvas.shape[1]):
            pc.append(Polygon([(0, 32 * j), (height, 32 * j)], color='grey'))

        for (i, j), patches in np.ndenumerate(canvas):
            if patches is not None:
                for patch in patches:
                    transform = Affine2D().translate(32 * j, 32 * (array.shape[1] - i - 1))
                    patch.set_transform(transform)
                    pc.append(patch)

        pc = PatchCollection(pc, match_original=True)
        ax.add_collection(pc)

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.set_aspect('equal')

        return ax

    def setup(func):

        @functools.wraps(func)
        def wrap(self, color, highlight=False, **kwargs):
            patches = func(self, color, **kwargs)
            if patches is not None:
                if not isinstance(patches, list):
                    patches = [patches]

                if highlight:
                    overlay = Rectangle((0, 0), 32, 32, color='white', alpha=.75)
                    patches.append(overlay)

            return patches

        return wrap

    @setup
    def agent(self, color, state, highlight=False):
        xy = {'up': [(6, 4), (16, 28), (26, 4)],
              'right': [(4, 6), (28, 16), (4, 26)],
              'down': [(6, 28), (16, 4), (26, 28)],
              'left': [(28, 6), (4, 16), (28, 26)]}
        return Polygon(xy[state], color=color)

    @setup
    def empty(self, color, highlight=False):
        pass
        # return Rectangle((0, 0), 32, 32, color=color)

    @setup
    def wall(self, color, highlight=False):
        return Rectangle((0, 0), 32, 32, color=color)

    @setup
    def door(self, color, state, highlight=False):
        alpha = .25 if state == 'locked' else 0
        kwargs = dict(edgecolor=color, facecolor=colors.to_rgba(color, alpha))
        gap = 2
        if state == 'open':
            p = Polygon([
                (32, 32 - gap),
                (32, 32),
                (0, 32),
                (0, 32 - gap)
            ], **kwargs)
            return p

        patches = [Rectangle((0, 0), 32, 32, **kwargs),
                   Rectangle((gap, gap), 32 - 2 * gap, 32 - 2 * gap, **kwargs)]

        if state == 'locked':  # Draw key slot
            p = Polygon([(17.6, 16), (24, 16)], closed=False, **kwargs)
        else:  # Draw door handle
            p = Circle((24, 16), gap, **kwargs)
        patches.append(p)

        return patches

    @setup
    def key(self, color, highlight=False):
        return [
            # Vertical quad
            Rectangle((16, 4), 4, 18, color=color),
            # Teeth
            Rectangle((12, 11), 4, 2, color=color),
            Rectangle((12, 4), 4, 2, color=color),

            # outer key head
            Circle((18, 23), 6, color=color),

            # key hole
            Circle((18, 23), 2, color='black')
        ]

    @setup
    def ball(self, color, highlight=False):
        return Circle((16, 16), 10, color=color)

    @setup
    def box(self, color, highlight=False):
        kwargs = dict(linewidth=2, edgecolor=color, facecolor='black')
        return [
            Rectangle((4, 4), 24, 24, **kwargs),  # box
            Polygon([(4, 16), (28, 16)], **kwargs)  # horizontal line
            ]

    @setup
    def goal(self, color, highlight=False):
        off = 2 * np.sqrt(3)

        return [
            Rectangle((0, 0), 32, 32, color=color),
            # draw a star
            Polygon([(8, 16), (24, 16)], closed=False, color='black'),
            Polygon([(16 - off, 10), (16 + off, 22)], closed=False, color='black'),
            Polygon([(16 - off, 22), (16 + off, 10)], closed=False, color='black')
        ]

    @setup
    def lava(self, color, highlight=False):
        wave = np.array([
            (3.2, 16),
            (9.6, 12.8),
            (15, 16),
            (22.4, 12.8),
            (28.8, 16),
        ])
        kwargs = dict(closed=False, color='black', fill=False)
        return [
            # lava is originally red, but we override it with orange
            Rectangle((0, 0), 32, 32, color='orange'),

            # drawing the waves
            Polygon(wave - np.array([0, 6.4]), **kwargs),
            Polygon(wave, **kwargs),
            Polygon(wave + np.array([0, 6.4]), **kwargs)
        ]
