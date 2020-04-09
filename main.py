from backend.delphi_method import DelfiMethod

import pandas as pd
import numpy as np
import pyqtgraph as pg
import sys
#import qdarkstyle

from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QLabel, QSpinBox, QTabWidget, QApplication, QTextBrowser, QDoubleSpinBox, QTableView, QFrame, QListWidget
from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtCore import pyqtSlot

class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        data = np.array(data).tolist()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])



def connect(obj, func):
    if isinstance(obj, QSpinBox):
        obj.valueChanged.connect(func)
    else:
        obj.clicked.connect(func)
    return obj


PENS = {'low mean': pg.mkPen(color=(0, 255, 0)),
        'high mean': pg.mkPen(color=(0, 255, 0)),
        'mean': pg.mkPen(color=(255, 0, 0)),
        'Model -': pg.mkPen(color=(0, 0, 255)),
        'Model +': pg.mkPen(color=(0, 0, 255)),
        'Gauss +': pg.mkPen(color=(0, 255, 0)),
        'Gauss -': pg.mkPen(color=(0, 255, 0)),
        'Expert median': pg.mkPen(color=(0, 255, 255)),
        }


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.__initUI__()
        self.method = None


    def __initUI__(self):
        self.__calc_button = QPushButton('Calculate Results')
        self.__calc_button.clicked.connect(self.__button_press)



        self.main_df_1 = QTableView()
        self.main_df_2 = QTableView()
        self.main_df_3 = QTableView()
        main_info = QGridLayout()
        main_info.addWidget(self.main_df_1, 0, 0,-1,1)
        main_info.addWidget(self.main_df_2, 0, 1,)
        main_info.addWidget(self.main_df_3, 1, 1,)
        main_info_frame = QFrame()
        main_info_frame.setLayout(main_info)


        self.cases_list = QListWidget()
        self.criteria_list = QListWidget()
        self.experts_list = QListWidget()
        self.plot_button = QPushButton('Plot')
        self.plot_button.clicked.connect(self.__plot_button_press)
        self.graphics_tabs = QTabWidget()
        self.experts_graph = pg.PlotWidget()
        self.data_tabs = QTabWidget()
        self.text = QTextBrowser()
        plot_tab = QGridLayout()
        plot_tab.setVerticalSpacing(20)
        plot_tab.setHorizontalSpacing(20)
        plot_tab.addWidget(self.cases_list, 0, 0)
        plot_tab.addWidget(self.criteria_list, 0, 1, )
        plot_tab.addWidget(self.experts_list, 0, 2, )
        plot_tab.addWidget(self.plot_button, 0, 3, )
        plot_tab.addWidget(self.graphics_tabs, 1, 0, 1, 3)
        plot_tab.addWidget(self.experts_graph, 1, 3, )
        plot_tab.addWidget(self.data_tabs, 2, 0, 1, 3)
        plot_tab.addWidget(self.text, 2, 3, )
        plot_tab_frame = QFrame()
        plot_tab_frame.setLayout(plot_tab)




        menu = QTabWidget()
        menu.addTab(self.__calc_button, 'input')
        menu.addTab(main_info_frame, 'main_info')
        menu.addTab(plot_tab_frame, 'plot_tab')


        main_layout = QGridLayout()
        main_layout.addWidget(menu, 0, 0)
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

    def get_plot_params(self):
        params = {
            'case': self.cases_list.currentItem(),
            'criteria': self.criteria_list.currentItem(),
            'expert': self.experts_list.currentItem(),
        }
        return params


    @pyqtSlot()
    def __button_press(self):
        self.method = DelfiMethod.initialize_from_py_file('input_data/constants.py')

        main_info = self.method.get_main_info()
        self.main_df_1.setModel(TableModel(main_info['Results of experts estimation by Delphi method'].reset_index().astype(str)))
        self.main_df_2.setModel(TableModel(main_info['Agreed quantitative estimates of the expert survey by Delphi method(numbers)'].data.reset_index()))
        self.main_df_3.setModel(TableModel(main_info['Agreed quantitative estimates of the expert survey by Delphi method(values)'].data.reset_index()))

        self.cases_list.addItems(main_info['cases'])
        self.criteria_list.addItems(main_info['criteria'])
        self.experts_list.addItems([str(el) for el in main_info['experts']])
        return



    def plot_grapg(self, plot_widget, data):
        if 'line_intensity' in data.keys():
            data.pop('line_intensity')
        if 'line_mask' in data.keys():
            data.pop('line_mask')

        if 'index' in data.keys():
            idx = data.pop('index')
            for key, value in data.items():
                if (key in PENS.keys()) or (key[:-2] in PENS.keys()):
                    plot_widget.plot(idx, value, pen=PENS[key])
                else:
                    plot_widget.plot(idx, value)
        else:
            print('non_index')
            for key, value in data.items():
                idx = list(range(len(value)))
                if (key in PENS.keys()) or (key[:-2] in PENS.keys()):
                    plot_widget.plot(idx, value, pen=PENS[key])
                else:
                    plot_widget.plot(idx, value)


    @pyqtSlot()
    def __plot_button_press(self):
        if not self.method:
            return

        params = self.get_plot_params()
        if any([p is None for p in params.values()]):
            return

        expert_plot = self.method.get_expert_info(params['case'].text(), params['criteria'].text(),int(params['expert'].text()))
        self.plot_grapg(self.experts_graph, expert_plot['Interval estimate of expert'])
        QApplication.processEvents()

        response = self.method.get_case_criteria_info(params['case'].text(), params['criteria'].text())

        self.text.setText(response['Cluster main information'])
        QApplication.processEvents()

        data_names = ['Input expert estimations', 'Expert interval estimates', 'Heatmap of experts estimates distances']
        self.data_tabs.clear()
        for name in data_names:
            data = pd.DataFrame(response[name]).round(2).astype(str)
            table = QTableView()
            table.setModel(TableModel(data))
            self.data_tabs.addTab(table, name)
            QApplication.processEvents()

        plot_names = ['Point Estimate',
            'Interval estimate of the mean of interval estimates',
            'Interval integrated expert estimates',
            'Discrete interval gaussian density',
            'Estimate of experts by the lowest and the highest quality value',
            'Median interval estimate',
            'Experts estimates that are/are not in confidence interval']
        self.graphics_tabs.clear()
        for name in plot_names:
            plot_widget = pg.PlotWidget()
            self.plot_grapg(plot_widget, response[name])
            self.graphics_tabs.addTab(plot_widget, name)
            QApplication.processEvents()
        return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    #app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
