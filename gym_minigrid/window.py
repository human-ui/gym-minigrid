import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTextEdit
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QFrame


class Window(QMainWindow):
    """
    Simple application window to render the environment into
    """

    KEYS = {
        Qt.Key_Left: 'LEFT',
        Qt.Key_Right: 'RIGHT',
        Qt.Key_Up: 'UP',
        Qt.Key_Down: 'DOWN',
        Qt.Key_Space: 'SPACE',
        Qt.Key_Return: 'RETURN',
        Qt.Key_Alt: 'ALT',
        Qt.Key_Control: 'CTRL',
        Qt.Key_PageUp: 'PAGE_UP',
        Qt.Key_PageDown: 'PAGE_DOWN',
        Qt.Key_Backspace: 'BACKSPACE',
        Qt.Key_Escape: 'ESCAPE'
    }

    def __init__(self, env, renderer):
        self.app = QApplication([])
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

        self.setText(self.env.mission)
        self.render()

        # Show the application window
        self.show()
        self.setFocus()
        sys.exit(self.app.exec_())

    def processEvents(self):
        self.app.processEvents()

    def setPixmap(self, pixmap):
        self.imgLabel.setPixmap(pixmap)

    def setText(self, text):
        self.missionBox.setPlainText(text)

    def keyPressEvent(self, e):
        keyName = self.KEYS.get(e.key())
        if keyName is None:
            return
        self.keyDownCb(keyName)

    def render(self):
        self.setPixmap(self.renderer.pixmap())
        self.processEvents()

    def keyDownCb(self, keyName):
        if keyName == 'BACKSPACE':
            self.env.reset()
            self.setText(self.env.mission)
            print(self.env)
            self.render()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = self.env.actions.index('left')
        elif keyName == 'RIGHT':
            action = self.env.actions.index('right')
        elif keyName == 'UP':
            action = self.env.actions.index('forward')

        elif keyName == 'SPACE':
            action = self.env.actions.index('toggle')
        elif keyName == 'PAGE_UP':
            action = self.env.actions.index('pickup')
        elif keyName == 'PAGE_DOWN':
            action = self.env.actions.index('drop')

        elif keyName == 'RETURN':
            action = self.env.actions.index('done')

        else:
            print('unknown key %s' % keyName)
            return

        _, _, done, _ = self.env.step(action)

        if done:
            self.env.reset()
            self.setText(self.env.mission)
        self.render()
        return action
