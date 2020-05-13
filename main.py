import sys
import qdarkstyle

from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QHeaderView, QSpinBox, QTabWidget, QApplication, \
    QTextBrowser, QLineEdit, QTableView, QComboBox, QLabel
from PyQt5.QtCore import Qt, QAbstractTableModel, pyqtSlot

from backend import GraphProcessor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def setContent(self, data):
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])


def connect(obj, func):
    if isinstance(obj, QSpinBox):
        obj.valueChanged.connect(func)
    else:
        obj.clicked.connect(func)
    return obj


class QTabWidgetFixed(QTabWidget):
    def showEvent(self, a0):
        try:
            return super().showEvent(a0)
        finally:
            if Qt.WindowState == Qt.WindowMaximized:
                self.setFixedSize(self.size())
                # pass


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.graph_processor = GraphProcessor()
        self.__initUI__()

        self.__stop_calc = False

    def __initUI__(self):
        self.figure = self.graph_processor.plot_graph_on_figure()
        self.graph_canvas = FigureCanvas(self.figure)

        self.plot_figure = self.graph_processor.plot_scenario()
        self.plot_canvas = FigureCanvas(self.plot_figure)

        self.table = QTableView()
        self.table.setModel(TableModel(self.graph_processor.df))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.data_tabs = QTabWidgetFixed()
        self.data_tabs.setMinimumHeight(400)
        self.data_tabs.setMinimumWidth(700)
        self.data_tabs.addTab(self.graph_canvas, 'Граф')
        self.data_tabs.addTab(self.table, 'Таблиця')
        self.data_tabs.addTab(self.plot_canvas, 'Імпульсне моделювання')

        self.scenario_len_input = QLineEdit()
        self.scenario_len_input.setText('20')
        self.scenario_node = QComboBox()
        self.impulse_input = QLineEdit()
        self.impulse_time_input = QLineEdit()
        self.add_impulse_button = QPushButton('Додати імпульс')
        self.add_impulse_button.clicked.connect(self.add_impulse)
        self.impulses_output = QLabel('')

        self.restore_button = QPushButton('Відновити початкові значення')
        self.restore_button.clicked.connect(self.restore_default_values)

        self.new_node = QLineEdit()
        self.node_value = QLineEdit()
        self.node_value.setText('1')

        self.edge_node_1 = QComboBox()
        self.edge_node_2 = QComboBox()

        for node_box in (self.edge_node_1, self.edge_node_2, self.scenario_node):
            node_box.addItems(self.graph_processor.get_factor_names())

        self.edge_value = QLineEdit()

        self.add_node_button = QPushButton('Додати фактор')
        self.add_node_button.clicked.connect(self.add_node)

        self.delete_node_button = QPushButton('Вилучити фактор')
        self.delete_node_button.clicked.connect(self.delete_node)

        self.add_edge_button = QPushButton("Додати зв'язок")
        self.add_edge_button.clicked.connect(self.add_edge)

        self.text_output = QTextBrowser()
        self.text_output.setMinimumHeight(300)
        self.text_output.setMinimumWidth(400)

        self._update_text()

        main_layout = QGridLayout()
        main_layout.setVerticalSpacing(20)
        main_layout.setHorizontalSpacing(20)

        main_layout.addWidget(self.data_tabs, 0, 0, 3, 3)

        sub_layout = QGridLayout()
        sub_layout.setVerticalSpacing(50)
        sub_layout.addWidget(QLabel('Довжина сценарію'), 0, 0)
        sub_layout.addWidget(self.scenario_len_input, 0, 1, 1, 2)

        sub_layout.addWidget(self.scenario_node, 1, 0)
        sub_layout.addWidget(self.impulse_input, 1, 1)
        sub_layout.addWidget(self.impulse_time_input, 1, 2)

        sub_layout.addWidget(QLabel('Фактор'), 2, 0)
        sub_layout.addWidget(QLabel('Значення'), 2, 1)
        sub_layout.addWidget(QLabel('Час застосування'), 2, 2)

        sub_layout.addWidget(self.add_impulse_button, 3, 0, 1, 3)
        sub_layout.addWidget(self.impulses_output, 4, 0, 1, 3)
        main_layout.addLayout(sub_layout, 0, 4, 3, 1, Qt.AlignCenter)

        main_layout.addWidget(self.restore_button, 2, 4)

        sub_layout = QGridLayout()
        sub_layout.addWidget(QLabel('Назва фактору'), 0, 0)
        sub_layout.addWidget(self.new_node, 0, 1)
        sub_layout.addWidget(QLabel('Початкове значення'), 1, 0)
        sub_layout.addWidget(self.node_value, 1, 1)
        main_layout.addLayout(sub_layout, 3, 0)

        main_layout.addWidget(self.add_node_button, 4, 0)
        main_layout.addWidget(self.delete_node_button, 5, 0)

        sub_layout = QGridLayout()
        sub_layout.addWidget(QLabel('Перший фактор'), 0, 0)
        sub_layout.addWidget(self.edge_node_1, 0, 1)
        sub_layout.addWidget(QLabel('Другий фактор'), 1, 0)
        sub_layout.addWidget(self.edge_node_2, 1, 1)
        sub_layout.addWidget(QLabel("Значення зв'язку"), 2, 0)
        sub_layout.addWidget(self.edge_value, 2, 1)
        main_layout.addLayout(sub_layout, 3, 1, 2, 1)

        main_layout.addWidget(self.add_edge_button, 5, 1, 1, 2)

        main_layout.addWidget(self.text_output, 3, 4, 3, 1)

        self.setLayout(main_layout)

    def _update(self):
        self._update_graph()
        self._update_text()
        self.table.setModel(TableModel(self.graph_processor.df))

    def _update_text(self):
        self.impulses_output.clear()
        self.text_output.clear()
        self.text_output.append(self.graph_processor.graph_info())

    def _update_graph(self):
        self.graph_processor.plot_graph_on_figure(self.figure)
        self.figure.canvas.draw()

        self.graph_processor.plot_scenario(self.plot_figure)
        self.plot_figure.canvas.draw()

    def _read_scenario_input(self):
        text = self.scenario_len_input.text()
        self.graph_processor.scenario_impulses = [line.split(sep=',') for line in text.splitlines(keepends=False)]

    @pyqtSlot()
    def restore_default_values(self):
        self.graph_processor.reset()
        self.graph_processor.reset_scenario()
        self._update()

    @pyqtSlot()
    def add_node(self):
        self.graph_processor.add_node(self.new_node.text(), float(self.node_value.text()))
        for node_box in (self.edge_node_1, self.edge_node_2, self.scenario_node):
            node_box.addItem(self.new_node.text())
        self.graph_processor.reset_scenario()
        self._update()

    @pyqtSlot()
    def delete_node(self):
        self.graph_processor.delete_node(self.new_node.text())
        for node_box in (self.edge_node_1, self.edge_node_2, self.scenario_node):
            node_box.removeItem(node_box.findText(self.new_node.text()))
        self.graph_processor.reset_scenario()
        self._update()

    @pyqtSlot()
    def add_edge(self):
        value = float(self.edge_value.text())
        if self.edge_value:
            self.graph_processor.add_edge(self.edge_node_1.currentText(), self.edge_node_2.currentText(), value)
        else:
            self.graph_processor.delete_edge(self.edge_node_1.currentText(), self.edge_node_1.currentText())
        self.graph_processor.reset_scenario()
        self._update()

    @pyqtSlot()
    def add_impulse(self):
        self.graph_processor.add_impulse(
            node=self.scenario_node.currentText(),
            value=float(self.impulse_input.text()),
            time=int(self.impulse_time_input.text()))
        self.impulses_output.setText(
            '\n'.join(f'час: {time}, імпульси: {impulses}'
                      for time, impulses in self.graph_processor.scenario_impulses.items() if impulses))
        self.graph_processor.scenario_len = int(self.scenario_len_input.text())
        self.graph_processor.process_scenario()
        self._update_graph()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main = MainWindow()
    main.showMaximized()
    sys.exit(app.exec_())
