import sys
from PyQt5.QtWidgets import *
from model.model import Model 
from screens.mainPage import MainPage
import shared

app = QApplication(sys.argv)
main_window = MainPage()
shared.model = Model()
shared.stack_w = QStackedWidget()
shared.stack_w.addWidget(main_window)
shared.stack_w.show()
app.exec_()