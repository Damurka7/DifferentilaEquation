import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QDialog, QVBoxLayout, QHBoxLayout, \
    QPlainTextEdit, QLabel
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

y_o = 1
x_o = -4
x_1 = 4
chosenMethod = 0



class InitialValueProblem:
    x_coordinates = []
    y_approximate = []
    y_exact = []
    GTE = []
    LTE = []

    def __init__(self, method_name, y_o, x_o, x_1, h):
        self.x_coordinates = []
        self.y_approximate = []
        self.y_exact = []
        self.data = pd.DataFrame()
        self.method_name = method_name
        self.y_o = y_o
        self.x_o = x_o
        self.x_1 = x_1
        self.h = h
        self.x_coordinates = np.arange(self.x_o, self.x_1 + self.h, self.h)  # array with x-coordinates
        self.y_exact = np.array([self.get_y_exact(i) for i in self.x_coordinates])  # array with exact y-coordinates
        self.y_approximate.append(self.y_exact[0])
        for i in range(1, len(self.x_coordinates)):
            self.y_approximate.append(self.calc_y(self.x_coordinates[i - 1], self.y_approximate[i - 1]))
        self.GTE = []
        for i in range(len(self.x_coordinates)):
           self.GTE.append(abs(self.y_exact[i] - self.y_approximate[i]))
        self.LTE = []
        self.LTE.append(0.0)
        for i in range(1, len(self.x_coordinates)):
            self.LTE.append(abs(self.y_exact[i] - self.calc_y(self.x_coordinates[i - 1], self.y_exact[i - 1])))
        self.data = pd.DataFrame(
            {'x_coordinates': self.x_coordinates, 'Y-exact': self.y_exact, 'Y-' + str(self.method_name): self.y_approximate,
             'LTE': self.LTE, 'GTE': self.GTE})
        print(self.data)

    def get_y_exact(self, x):
        return math.e ** (-x) / (1 - math.e ** (4 + x) + math.e ** (8 + x))

    def y_differential(self, x, y):
        return y * y * math.e ** x - 2 * y

    def calc_y(self, x, y):  # to override this method in subclasses
        return None


class ImprovedEulerMethod(InitialValueProblem):
    def __init__(self, y_o, x_o, x_1, h):
        super(ImprovedEulerMethod, self).__init__("Improved Euler Method", y_o, x_o, x_1, h)

    def calc_y(self, x, y):
        return y + self.h * self.y_differential(x + self.h / 2, y + self.h / 2 * self.y_differential(x, y))


class EulerMethod(InitialValueProblem):
    def __init__(self, y_o, x_o, x_1, h):
        super(EulerMethod, self).__init__("Euler Method", y_o, x_o, x_1, h)

    def calc_y(self, x, y):
        return y + self.h * self.y_differential(x, y)


class RungeKuttaMethod(InitialValueProblem):
    def __init__(self, y_o, x_o, x_1, h):
        super(RungeKuttaMethod, self).__init__("Runge-Kutta", y_o, x_o, x_1, h)

    def calc_y(self, x, y):
        k1 = self.y_differential(x, y)
        k2 = self.y_differential(x + self.h / 2, y + self.h * k1 / 2)
        k3 = self.y_differential(x + self.h / 2, y + self.h * k2 / 2)
        k4 = self.y_differential(x + self.h, y + self.h * k3)
        return y + self.h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class Window(QDialog):
    h = 0.1
    method = EulerMethod( y_o, x_o, x_1, h)
    switcher = 0  # 0 - Euler, 1 - Improved Euler, 2 - Runge-Kutta

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        # graph of function
        self.figure = plt.figure("function")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # graph of GTE
        self.gte_fig = plt.figure("GTE")
        self.gte_canvas = FigureCanvas(self.gte_fig)
        self.gte_toolbar = NavigationToolbar(self.gte_canvas, self)
        self.plotGTE()

        # graph of LTE
        self.lte_fig = plt.figure("LTE")
        self.lte_canvas = FigureCanvas(self.lte_fig)
        self.lte_toolbar = NavigationToolbar(self.lte_canvas, self)
        self.plotLTE()
        self.plotEuler(self.h)

        # adding buttons and their logic
        self.button1 = QPushButton('\nPlot Euler\n')
        self.button1.clicked.connect(lambda : self.plotEuler(self.h))
        self.button2 = QPushButton('\nPlot Improved Euler\n')
        self.button2.clicked.connect(lambda : self.plotImprovedEuler(self.h))
        self.button3 = QPushButton('\nPlot Runge Kutta\n')
        self.button3.clicked.connect(lambda: self.plotRungeKutta(self.h))
        self.pushButton = QPushButton('\nEnter h:\n')
        self.pushButton.clicked.connect(self.takeinputs)

        self.label = QLabel("grid is :" + str(self.h))

        # adding all stuff to the application
        hbox = QHBoxLayout()
        layout = QVBoxLayout()

        toolbar_vl = QVBoxLayout()
        toolbar_vl.addWidget(self.toolbar)
        toolbar_vl.addWidget(self.canvas)
        hbox.addLayout(toolbar_vl)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.button1)
        button_layout.addWidget(self.button2)
        button_layout.addWidget(self.button3)
        button_layout.addWidget(self.label)
        button_layout.addWidget(self.pushButton)
        #button_layout.addWidget(self.text)
        hbox.addLayout(button_layout)

        error_box = QHBoxLayout()
        toolbar_vl = QVBoxLayout()
        toolbar_vl.addWidget(self.gte_toolbar)
        toolbar_vl.addWidget(self.gte_canvas)
        error_box.addLayout(toolbar_vl)

        toolbar_vl = QVBoxLayout()
        toolbar_vl.addWidget(self.lte_toolbar)
        toolbar_vl.addWidget(self.lte_canvas)
        error_box.addLayout(toolbar_vl)
        layout.addLayout(hbox)
        layout.addLayout(error_box)

        self.setLayout(layout)

    def takeinputs(self):
        grid = 0.1000
        grid, done = QtWidgets.QInputDialog.getDouble(
            self, 'Input Dialog', 'Enter grid size:', 0.1, 0, 5, 5)

        self.label.setText("grid is :" + str(self.h))
        if grid <=0 :
            self.label.setText("you have inserted a grid <= 0, please re-enter the grid (grid is default now):" + str(self.h))
            grid = 0.1
        self.h = grid
        if done:
            if self.switcher == 0:
                self.plotEuler(grid)
            elif self.switcher == 1:
                self.plotImprovedEuler(grid)
            else:
                self.plotRungeKutta(grid)
            self.plotLTE()
            self.plotGTE()

    def plotLTE(self):
        self.lte_fig.clear()
        # create an axis
        ax = self.lte_fig.add_subplot(111)

        # discards the old graph

        # plot data
        ax.plot(self.method.x_coordinates, self.method.LTE, label="LTE graph")
        ax.legend()
        ax.set(xlabel='x-axis', ylabel='y-axis', title="LTE")

        ax.grid()
        # refresh canvas
        self.lte_canvas.draw()

    def plotGTE(self):
        self.gte_fig.clear()
        # create an axis
        ax = self.gte_fig.add_subplot(111)

        # discards the old graph

        # plot data
        ax.plot(self.method.x_coordinates, self.method.GTE, label="GTE graph")
        ax.legend()
        ax.set(xlabel='x-axis', ylabel='y-axis', title="GTE")

        ax.grid()
        # refresh canvas
        self.gte_canvas.draw()

    def plotEuler(self, h):
        self.method = EulerMethod(y_o, x_o, x_1, h)
        self.switcher = 0
        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph

        # plot data
        ax.plot(self.method.x_coordinates, self.method.y_exact, label= "exact graph")
        ax.plot(self.method.x_coordinates, self.method.y_approximate, label="approximation")
        ax.legend()
        ax.set(xlabel='x-axis', ylabel='y-axis', title=self.method.method_name)

        ax.grid()
        # refresh canvas
        self.plotGTE()
        self.plotLTE()
        self.canvas.draw()

    def plotImprovedEuler(self, h):
        self.method = ImprovedEulerMethod(y_o, x_o, x_1, h)
        self.switcher = 1
        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph

        # plot data
        ax.plot(self.method.x_coordinates, self.method.y_exact, label= "exact graph")
        ax.plot(self.method.x_coordinates, self.method.y_approximate, label="approximation")
        ax.legend()
        ax.set(xlabel='x-efewfe', ylabel='y-efwef',
                    title=self.method.method_name)
        ax.grid()
        # refresh canvas
        self.plotGTE()
        self.plotLTE()
        self.canvas.draw()

    def plotRungeKutta(self, h):
        self.method = RungeKuttaMethod(y_o, x_o, x_1, h)
        self.switcher = 2
        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)



        # plot data
        ax.plot(self.method.x_coordinates, self.method.y_exact, label= "exact graph")
        ax.plot(self.method.x_coordinates, self.method.y_approximate, label="approximation")
        ax.legend()
        ax.set(xlabel='x-axis', ylabel='y-axis',
                    title=self.method.method_name)
        ax.grid()
        # refresh canvas
        self.plotGTE()
        self.plotLTE()
        self.canvas.draw()


def application():
    app = QApplication(sys.argv)
    main = Window()
    main.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    application()