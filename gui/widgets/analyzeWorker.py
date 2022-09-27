from PyQt5.QtCore import QObject, QThread, pyqtSignal
import shared

class Worker(QObject):
    
    processed = pyqtSignal(list)
    finished = pyqtSignal()
    
    def setFileSource(self, src):
        self.src = src

    def run(self):
        """Long-running task."""
        res = shared.model.process(self.src)
        self.processed.emit(res)
        self.finished.emit()