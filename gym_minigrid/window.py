import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('MacOSX')

from gym_minigrid import render


class Window(object):
    """
    Simple application window to render the environment into
    """

    KEYS = ['left', 'right', 'up', 'pageup', 'pagedown', ' ', 'enter']

    def __init__(self, env):
        self.env = env
        self.renderer = render.Matplotlib()

        # Create the figure and axes
        self.fig, self.ax = plt.subplots()

        # Show the env name in the window title
        self.fig.canvas.set_window_title(env.spec.id)

        # Turn off x/y axis numbering/ticks
        # self.ax.set_xticks([], [])
        # self.ax.set_yticks([], [])

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)
        # Keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', self.key_handler)

    def show(self, block=True):
        self.render()

        # If not blocking, trigger interactive mode
        # if not block:
        #     plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def render(self):
        array = self.env.grid.asarray()[0]
        plt.cla()
        self.renderer.render(array, ax=self.ax)
        self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        # plt.pause(0.001)

    def key_handler(self, event):
        action = event.key

        if action == 'escape':
            plt.close()
            return

        if action == 'backspace':
            self.env.reset()
            self.render()
            return

        try:
            action_idx = self.KEYS.index(action)
        except ValueError:
            print(f'unknown action {action}')

        obs, reward, done, info = self.env.step(np.repeat(action_idx, self.env.n_envs))

        if done[0]:
            self.env.reset()
        self.render()
