from cgitb import reset
from PyQt5.uic import loadUi
from pathlib import Path
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer, QMediaPlaylist
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QUrl
from PyQt5.QtCore import QThread
from widgets.analyzeWorker import Worker
import shared
import multiprocessing

class Player(QMainWindow):
    def __init__(self):
        super(Player, self).__init__()
        loadUi("screens/player.ui", self)
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setVideoOutput(self.video_player)
        self.fname = None

        self.reset()
        self.back_b.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.back_b.clicked.connect(self.onBack)
        self.select_video_b.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.select_video_b.clicked.connect(self.onSelectVideoDialog)
        self.analyze_b.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.analyze_b.clicked.connect(self.analyze)

        # self.fname = None
        self.onSelectVideoDialog()  # anti pattern but works

    def reset(self):
        self.label.setText("Open a video and start analyzing.")
        self.labels = None
        try:
            self.mediaPlayer.positionChanged.disconnect()
        except:
            pass

    def onBack(self):
        this_widget = shared.stack_w.currentWidget()
        shared.stack_w.removeWidget(this_widget)
        this_widget.deleteLater()

    def onSelectVideoDialog(self):
        home_dir = str(Path.home())
        new_fname = QFileDialog.getOpenFileName(self, "Open file", home_dir)
        if new_fname[0] == "":
            return

        self.fname = new_fname
        self.reset()
        self.playlist = QMediaPlaylist()
        self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(self.fname[0])))
        self.playlist.setCurrentIndex(1)
        self.playlist.setPlaybackMode(QMediaPlaylist.CurrentItemInLoop)

        self.mediaPlayer.setPlaylist(self.playlist)
        self.mediaPlayer.play()

    def analyze(self):
        self.thread = QThread()
        self.worker = Worker()
        self.worker.setFileSource(self.fname[0])
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self.thread.deleteLater)
        # self.thread.finished.connect(lambda: )

        self.worker.processed.connect(self.onAnalyzingDone)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        # self.worker.aborted.conect(self.worker)

        self.thread.start()
        self.analyze_b.setEnabled(False)
        self.label.setText("Processing...")

    def output(self):
        self.i = 0
        self.mediaPlayer.stop()
        self.mediaPlayer.positionChanged.connect(self.onPositionChanged)
        self.mediaPlayer.setNotifyInterval(int(1000 / shared.model.frames_per_second))
        self.mediaPlayer.play()

    def onPositionChanged(self):
        self.label.setText(self.labels[self.i])
        self.i = min(self.i + 1, len(self.labels) - 1)

    def onAnalyzingDone(self, labelList):
        self.labels = labelList
        self.output()
        self.analyze_b.setEnabled(True)
