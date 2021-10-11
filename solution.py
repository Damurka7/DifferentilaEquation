import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets  import QApplication, QWidget, QMainWindow
from PyQt5 import QtWidgets

y_o = 1
x_o = -4
x_1 = 4

class InitialValueProblem:
    x_coordinates = []
    y_approximate = []
    y_exact = []

    def __init__(self, method_name, y_o, x_o, x_1, h):
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
        GTE = []
        for i in range(len(self.x_coordinates)):
            GTE.append(abs(self.y_exact[i] - self.y_approximate[i]))
        LTE = []
        LTE.append(0.0)
        for i in range(1, len(self.x_coordinates)):
            LTE.append(abs(self.y_exact[i] - self.calc_y(self.x_coordinates[i - 1], self.y_exact[i - 1])))
        self.data = pd.DataFrame(
            {'x_coordinates': self.x_coordinates, 'Y-exact': self.y_exact, 'Y-' + str(self.method_name): self.y_approximate,
             'LTE': LTE, 'GTE': GTE})
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
        super(EulerMethod, self).__init__("EulerMethod", y_o, x_o, x_1, h)

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


class Canvas(FigureCanvas):
    def __init__(self, parent):
        fig, self.ax = plt.subplots(figsize=(5, 4), dpi=150)
        super().__init__(fig)
        self.setParent(parent)

        """ 
        Matplotlib Script
        """

        method = EulerMethod(y_o, x_o, x_1, 0.1)

        self.ax.plot(method.x_coordinates, method.y_exact, label="exact graph")
        self.ax.plot(method.x_coordinates, method.y_approximate, label="approximation")
        self.ax.legend()
        self.ax.set(xlabel='x-axis', ylabel='y-axis',
                    title=method.method_name)
        self.ax.grid()



class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(300, 300, 1000, 800)
        self.setWindowTitle('Application')


        text = QtWidgets.QLabel(self)
        text.setText("Choose a  method")
        text.move(800, 100)
        text.adjustSize()

        chart = Canvas(self)



def application():
    app = QApplication(sys.argv)
    demo = AppDemo()



    demo.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    application()