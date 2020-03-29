from backend import IntegralAwareness


def get_outputs(i: int, j: int, time_start: int, time_stop: int, time_step: int, max_prob: float):
    factors = IntegralAwareness.default_instance(max_prob=max_prob, limit_t=time_stop)
    time_range = range(time_start, time_stop, time_step)
    i -= 1
    j -= 1
    #print(factors.plot_iter(i, j, time_range))
    return {
        'plot_dict': factors.get_plot_dict(i, j, time_range),
        'critical_time_range': factors.critical_time_range(i),
        'classification_df': factors.classification_df()
    }

'''
def run():
    print(get_outputs(3, 4, 0, 100, 1, 0.2))


if __name__ == '__main__':
    run()
'''


import pyqtgraph as pg
import sys
#import qdarkstyle

from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QLabel, QSpinBox, QTabWidget, QApplication, QTextBrowser, QDoubleSpinBox
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSlot


def connect(obj, func):
    if isinstance(obj, QSpinBox):
        obj.valueChanged.connect(func)
    else:
        obj.clicked.connect(func)
    return obj


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.__initUI__()

        self.__stop_calc = False

    def __initUI__(self):
        self.__i = QSpinBox(value=4)
        self.__i.setRange(0, 10)
        self.__j = QSpinBox(value=7)
        self.__j.setRange(0, 10)
        self.__prob = QDoubleSpinBox(value=0.5, )
        self.__prob.setRange(0, 1)
        self.__prob.setSingleStep(0.1)
        self.__start = QSpinBox(value=0)
        self.__stop = QSpinBox(value=100)
        self.__step = QSpinBox(value=1)

        self.__calc_button = QPushButton('Calculate Results')
        self.__calc_button.clicked.connect(self.__button_press)

        self.text_output = QTextBrowser()
        self.graphics_tabs = QTabWidget()

        i_j_grid = QGridLayout()
        i_j_grid.setVerticalSpacing(5)
        i_j_grid.addWidget(QLabel('i'), 0, 0)
        i_j_grid.addWidget(QLabel('j'), 1, 0)
        i_j_grid.addWidget(self.__i, 0, 1)
        i_j_grid.addWidget(self.__j, 1, 1)

        prob_grid = QGridLayout()
        prob_grid.setVerticalSpacing(5)
        prob_grid.addWidget(QLabel('prob'), 0, 0)
        prob_grid.addWidget(self.__prob, 0, 1)

        t_grid = QGridLayout()
        t_grid.setVerticalSpacing(5)
        t_grid.addWidget(QLabel('t start'), 0, 0)
        t_grid.addWidget(QLabel('t stop'), 1, 0)
        t_grid.addWidget(QLabel('t step'),2, 0)
        t_grid.addWidget(self.__start, 0, 1)
        t_grid.addWidget(self.__stop, 1, 1)
        t_grid.addWidget(self.__step, 2, 1)

        menu_layout = QGridLayout()
        menu_layout.setHorizontalSpacing(50)
        menu_layout.addWidget(QLabel('I and J', alignment=Qt.AlignCenter), 0, 0)
        menu_layout.addWidget(QLabel('Prob', alignment=Qt.AlignCenter), 0, 1)
        menu_layout.addWidget(QLabel('Time', alignment=Qt.AlignCenter), 0, 2)
        menu_layout.addLayout(i_j_grid, 1, 0)
        menu_layout.addLayout(prob_grid, 1, 1)
        menu_layout.addLayout(t_grid, 1, 2)

        self.graphics_tabs.addTab(pg.PlotWidget(), "Result")
        self.__calc_button.setMaximumWidth(300)

        main_layout = QGridLayout()
        main_layout.setVerticalSpacing(20)
        main_layout.addLayout(menu_layout, 0, 0, 1, -1)

        main_layout.addWidget(self.graphics_tabs, 2, 0)
        main_layout.addWidget(self.text_output, 2, 1)
        main_layout.addWidget(self.__calc_button, 1, 1, alignment=Qt.AlignRight)

        self.setLayout(main_layout)

    def get_params(self):
        params = {
            'i': self.__i.value(),
            'j': self.__j.value(),
            'max_prob': self.__prob.value(),
            'time_start': self.__start.value(),
            'time_stop': self.__stop.value(),
            'time_step': self.__step.value(),
        }
        return params


    @pyqtSlot()
    def __button_press(self):
        params = self.get_params()
        results = get_outputs(**params)

        time = results['plot_dict'].pop('time')
        self.graphics_tabs.clear()
        for key, value in results['plot_dict'].items():
            plot_widget = pg.PlotWidget()
            plot_widget.plot(time, value)
            self.graphics_tabs.addTab(plot_widget, key)
        self.text_output.setText(results['classification_df'].to_string())
        QApplication.processEvents()





if __name__ == '__main__':
    app = QApplication(sys.argv)
    #app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
