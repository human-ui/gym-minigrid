import sys
import numpy as np

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPolygon, QTransform
from PyQt5.QtCore import QPoint, QRect

from gym_minigrid import window


COLORS = {
    'black': [0, 0, 0],
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'yellow': [255, 255, 0],
    'blue': [0, 0, 255],
    'magenta': [255, 0, 255],
    'cyan': [0, 255, 255],
    'grey': [127, 127, 127],
    'white': [255, 255, 255],
    'orange': [255, 128, 0],
}


def scale(func):
    def wrapper(self, *coords):
        coords = [c / 32 * self.cell_pixels for c in coords]
        return func(self, *coords)
    return wrapper


def draw(func):
    def wrapper(self, *args, **kwargs):
        self.push()
        func(self, *args, **kwargs)
        self.pop()
    return wrapper


class Renderer(object):

    def __init__(self, env, highlight=True, cell_pixels=32):
        self.env = env
        self.highlight = highlight
        self.cell_pixels = cell_pixels
        self.height = self.env.shape[0] * self.cell_pixels
        self.width = self.env.shape[1] * self.cell_pixels

        self.img = QImage(self.width, self.height, QImage.Format_RGB888)
        self.painter = QPainter()
        # transpose i,j into j,i
        self.world_transform = QTransform(0, 1, 0,
                                          1, 0, 0,
                                          0, 0, 1)

    def beginFrame(self):
        self.painter.begin(self.img)
        self.painter.setRenderHint(QPainter.Antialiasing, False)
        self.painter.setWorldTransform(self.world_transform)

    def endFrame(self):
        self.painter.end()

    def getPixmap(self):
        return QPixmap.fromImage(self.img)

    def getArray(self):
        """
        Get a numpy array of RGB pixel values.
        The array will have shape (height, width, 3)
        """
        numBytes = self.width * self.height * 3
        buf = self.img.bits().asstring(numBytes)
        output = np.frombuffer(buf, dtype='uint8')
        output = output.reshape((self.height, self.width, 3))

        return output

    def push(self):
        self.painter.save()

    def pop(self):
        self.painter.restore()

    def rotate(self, degrees):
        self.painter.rotate(degrees)

    @scale
    def translate(self, x, y):
        self.painter.translate(x, y)

    # def scale(self, x, y):
    #     self.painter.scale(x, y)

    def setLineColor(self, r, g, b, a=255):
        self.painter.setPen(QColor(r, g, b, a))

    def setColor(self, r, g, b, a=255):
        self.painter.setBrush(QColor(r, g, b, a))

    def setLineWidth(self, width):
        pen = self.painter.pen()
        pen.setWidthF(width)
        self.painter.setPen(pen)

    @scale
    def drawLine(self, x0, y0, x1, y1):
        self.painter.drawLine(x0, y0, x1, y1)

    @scale
    def drawCircle(self, x, y, radius):
        center = QPoint(x, y)
        self.painter.drawEllipse(center, radius, radius)

    @scale
    def getQPoint(self, x, y):
        return QPoint(x, y)

    def drawPolygon(self, points):
        """Takes a list of points (tuples) as input"""
        points = [self.getQPoint(*point) for point in points]
        self.painter.drawPolygon(QPolygon(points))

    def drawPolyline(self, points):
        """Takes a list of points (tuples) as input"""
        points = [self.getQPoint(*point) for point in points]
        self.painter.drawPolyline(QPolygon(points))

    def drawRect(self, x, y, height, width):
        self.painter.drawRect(self.getRect(x, y, height, width))

    @scale
    def getRect(self, x, y, height, width):
        return QRect(x, y, height, width)

    def fillRect(self, x, y, height, width, r, g, b, a=255):
        self.painter.fillRect(self.getRect(x, y, height, width), QColor(r, g, b, a))


class MiniGridImage(Renderer):

    def show(self):
        app = QApplication([])
        win = window.Window(env=self.env, renderer=self)
        win.show()
        sys.exit(app.exec_())

    def array(self):
        self._render()
        return self.getArray()

    def pixmap(self):
        self._render()
        return self.getPixmap()

    def _render(self):
        self.beginFrame()

        # draw default background
        self.setColor(*COLORS['black'])
        self.drawRect(0, 0, 32 * self.env.shape[0], 32 * self.env.shape[1])

        # draw gridlines
        self.setLineColor(*COLORS['grey'])
        self.setLineWidth(1)
        for i in range(self.env.shape[0]):
            self.drawLine(32 * i, 0, 32 * i, 32 * self.env.shape[1])
        for j in range(self.env.shape[1]):
            self.drawLine(0, 32 * j, 32 * self.env.shape[0], 32 * j)

        mask = self.env.visible()
        for pos, cell in np.ndenumerate(self.env.grid):
            with self:
                if cell.color != 'black':
                    self._setup(pos, cell.color)
                    self.drawRect(0, 0, 32, 32)

            if cell.entity is not None:
                try:
                    draw_object = getattr(self, cell.entity.type)
                except AttributeError:
                    print(f'Cell type {cell.entity.type} not recognized')

                with self:
                    self._setup(pos, cell.entity.color)
                    draw_object(cell.entity)

            if mask[pos] and pos != self.env.agent.pos and self.highlight:
                self.highlight_cell(pos)

        with self:
            self._setup(self.env.agent.pos, self.env.agent.color)
            self.agent()

        if self.highlight:
            self.highlight_cell(self.env.agent.pos)

        self.endFrame()

    def __enter__(self):
        """Support with-statement for the environment. """
        self.push()
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment. """
        self.pop()
        return False

    def _setup(self, pos, color, lineWidth=1):
        self.translate(*pos)
        self.setLineColor(*COLORS[color])
        self.setLineWidth(lineWidth)
        self.setColor(*COLORS[color])

    @draw
    def highlight_cell(self, pos):
        self.translate(*pos)
        self.fillRect(0, 0, 32, 32, *COLORS['white'], a=75)

    def translate(self, i, j):
        super().translate(32 * i, 32 * j)

    @draw
    def agent(self):
        self.translate(.5, .5)
        self.rotate(-90 * self.env.agent.dir)
        self.drawPolygon([
            (-10, -12),
            (0, 12),
            (10, -12)
        ])

    @draw
    def wall(self, obj):
        self.drawRect(0, 0, 32, 32)

    @draw
    def door(self, obj):
        self.setColor(*COLORS[obj.color], 50 if obj.is_locked else 0)
        gap = 2
        if obj.is_open:
            self.drawPolygon([
                (32, 32 - gap),
                (32, 32),
                (0, 32),
                (0, 32 - gap)
            ])
            return

        self.drawRect(0, 0, 32, 32)
        self.drawRect(gap, gap, 32 - 2 * gap, 32 - 2 * gap)

        if obj.is_locked:  # Draw key slot
            self.drawLine(16, 17.6, 16, 24)
        else:  # Draw door handle
            self.drawCircle(16, 24, gap)

    @draw
    def key(self, obj):
        # Vertical quad
        self.drawRect(10, 16, 18, 4)
        # Teeth
        self.drawRect(19, 12, 2, 4)
        self.drawRect(26, 12, 2, 4)

        # outer key head
        self.drawCircle(9, 18, 6)

        # key hole
        self.setLineColor(*COLORS['black'])
        self.setColor(*COLORS['black'])
        self.drawCircle(9, 18, 2)

    @draw
    def ball(self, obj):
        self.drawCircle(16, 16, 10)

    @draw
    def box(self, obj):
        self.setColor(*COLORS['black'])  # set inside to black
        self.setLineWidth(2)
        self.drawRect(4, 4, 24, 24)  # box
        self.drawLine(16, 4, 16, 28)  # horizontal line

    @draw
    def goal(self, obj):
        self.drawRect(0, 0, 32, 32)

    def lava(self, obj):
        # lava is originally red, but we override it with orange
        self.setLineColor(*COLORS['orange'])
        self.setColor(*COLORS['orange'])

        self.drawRect(0, 0, 32, 32)

        # drawing the waves
        self.setLineColor(*COLORS['black'])
        wave = np.array([
            (16, 3.2),
            (19.2, 9.6),
            (16, 15),
            (19.2, 22.4),
            (16, 28.8),
        ])
        self.drawPolyline(wave - np.array([6.4, 0]))  # below
        self.drawPolyline(wave)  # middle
        self.drawPolyline(wave + np.array([6.4, 0]))  # above
