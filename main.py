def backend(params):
    return None


import numpy as np

import pyqtgraph as pg
import sys
#import qdarkstyle

from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QLabel, QSpinBox, QTabWidget, QApplication, QTextBrowser, QLineEdit, QTableView
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


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.__initUI__()

        self.__stop_calc = False

    def __initUI__(self):
        self.graph = pg.PlotWidget()
        self.table = QTableView()
        grapg_and_table = QTabWidget()
        grapg_and_table.setMinimumHeight(400)
        grapg_and_table.setMinimumWidth(700)
        grapg_and_table.addTab(self.graph, "Graph")
        grapg_and_table.addTab(self.table, 'table')

        self.new_scenario_input = QLineEdit()
        self.new_scenario_button = QPushButton('new_scenario')
        self.new_scenario_button.clicked.connect(self.new_scenario_button_func)
        self.restore_default_values_button = QPushButton('restore_default_values')
        self.restore_default_values_button.clicked.connect(self.restore_default_values_func)

        self.node_name_input_1 = QLineEdit()
        self.node_name_input_2 = QLineEdit()
        self.node_name_input_3 = QLineEdit()
        self.bound_value_input = QLineEdit()

        self.add_node_button = QPushButton('add_node')
        self.add_node_button.clicked.connect(self.add_node_func)

        self.delete_node_button = QPushButton('delete_node')
        self.delete_node_button.clicked.connect(self.delete_node_func)

        self.apply_bound_value_button = QPushButton('apply_bound_value')
        self.apply_bound_value_button.clicked.connect(self.apply_bound_value_func)

        self.text_output = QTextBrowser()
        self.text_output.setMinimumHeight(300)
        self.text_output.setMinimumWidth(400)

        main_layout = QGridLayout()
        main_layout.setVerticalSpacing(20)
        main_layout.setHorizontalSpacing(20)

        main_layout.addWidget(grapg_and_table, 0, 0, 3, 3)

        main_layout.addWidget(self.new_scenario_input, 0, 4,)
        main_layout.addWidget(self.new_scenario_button, 1, 4,)
        main_layout.addWidget(self.restore_default_values_button, 2, 4)

        main_layout.addWidget(self.node_name_input_1, 3, 0)
        main_layout.addWidget(self.add_node_button, 4, 0,)
        main_layout.addWidget(self.delete_node_button, 5, 0,)

        main_layout.addWidget(self.node_name_input_2, 3, 1,)
        main_layout.addWidget(self.node_name_input_3, 3, 2,)
        main_layout.addWidget(self.bound_value_input, 4, 1, 1, 2)
        main_layout.addWidget(self.apply_bound_value_button, 5, 1, 1, 2)

        main_layout.addWidget(self.text_output, 3, 4, 3, 1)

        self.setLayout(main_layout)

    def get_params(self):
        params = {
            'new_scenario_input': self.new_scenario_input.text(),
            'node_name_input_1': self.node_name_input_1.text(),
            'node_name_input_2': self.node_name_input_2.text(),
            'node_name_input_3': self.node_name_input_3.text(),
            'bound_value_input': self.bound_value_input.text(),
        }
        return params

    @pyqtSlot()
    def new_scenario_button_func(self):
        params = self.get_params()
        self.text_output.setText(str(params))
        return None

    @pyqtSlot()
    def restore_default_values_func(self):
        return None

    @pyqtSlot()
    def add_node_func(self):
        return None

    @pyqtSlot()
    def delete_node_func(self):
        return None

    @pyqtSlot()
    def apply_bound_value_func(self):
        return None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    #app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
