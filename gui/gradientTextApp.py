from PyQt5 import QtWidgets, QtCore, QtGui

class Widget(QtWidgets.QWidget):
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        font = QtGui.QFont("Arial", 72)
        painter.setFont(font)
        rect = self.rect()
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.topRight())
        gradient.setColorAt(0, QtCore.Qt.red)
        gradient.setColorAt(1, QtCore.Qt.blue)
        pen = QtGui.QPen()
        pen.setBrush(gradient)
        painter.setPen(pen)
        painter.drawText(QtCore.QRectF(rect), "Hello world", QtGui.QTextOption(QtCore.Qt.AlignCenter))

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = Widget()
    widget.show()
    app.exec()