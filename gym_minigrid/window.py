import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTextEdit
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QFrame


class Window(QMainWindow):
    """
    Simple application window to render the environment into
    """

    KEYS = {
        Qt.Key_Left: 'left',
        Qt.Key_Right: 'right',
        Qt.Key_Up: 'forward',
        Qt.Key_Space: 'toggle',
        Qt.Key_PageUp: 'pickup',
        Qt.Key_PageDown: 'drop',
        Qt.Key_Return: 'done',
        Qt.Key_Backspace: 'reset',
        Qt.Key_Escape: 'done'
    }

    def __init__(self, env, renderer):
        super().__init__()

        self.env = env
        self.renderer = renderer

        self.setWindowTitle('MiniGrid Gym Environment')

        # Image label to display the rendering
        self.imgLabel = QLabel()
        self.imgLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        # Text box for the mission
        self.missionBox = QTextEdit()
        self.missionBox.setReadOnly(True)
        self.missionBox.setMinimumSize(400, 100)

        self.missionBox.setPlainText(self.env.mission)

        # Center the image
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.imgLabel)
        hbox.addStretch(1)

        # Arrange widgets vertically
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.missionBox)

        # Create a main widget for the window
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
        mainWidget.setLayout(vbox)

    def show(self):
        self.render()
        super().show()
        self.setFocus()

    def keyPressEvent(self, e):
        action = self.KEYS.get(e.key())
        if action is None:
            return
        self.keyDownCb(action)

    def reset(self):
        self.env.reset()
        self.missionBox.setPlainText(self.env.mission)

    def render(self):
        self.imgLabel.setPixmap(self.renderer.pixmap())
        QApplication.processEvents()

    def keyDownCb(self, action):
        if action == 'reset':
            self.reset()
            self.render()
            return

        if action == 'done':
            sys.exit(0)

        try:
            action_idx = self.env.actions.index(action)
        except ValueError:
            print(f'unknown action {action}')

        obs, reward, done, reset_mask = self.env.step(action_idx)

        if done:
            self.reset()
        self.render()
        return obs, reward, done, reset_mask
