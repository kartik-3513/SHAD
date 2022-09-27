from inspect import stack
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from .player import Player
import shared


class MainPage(QMainWindow):
    def __init__(self):
        super(MainPage, self).__init__()
        loadUi("screens/mainScreen.ui", self)
        self.start_b.clicked.connect(self.onClickMainVideo)
        self.start_b.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def onClickMainVideo(self):
        player = Player()
        shared.stack_w.addWidget(player)
        shared.stack_w.setCurrentIndex(shared.stack_w.currentIndex() + 1)
