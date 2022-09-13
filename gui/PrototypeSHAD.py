import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

class MainPage(QMainWindow):
  def __init__(self):
    super(MainPage, self).__init__()
    loadUi("mainScreen.ui", self)
  
app = QApplication(sys.argv)
main_window = MainPage()
stack_w = QtWidgets.QStackedWidget()
stack_w.addWidget(main_window)
stack_w.show()
app.exec_()