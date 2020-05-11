from collections import defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches
import networkx as nx

np.set_printoptions(precision=3)


class GraphProcessor:
    def __init__(self):
        self.url = 'https://docs.google.com/spreadsheets/d/1WM2UkpA-0DqowowpkF7mck6d1tyFJNoWaSM7YsG6vRc' \
                   '/export?gid={gid}&format=csv'
        self.df = self.read_df()
        self.start_values = self.read_start_values()
        self.graph = self.create_graph()
        self.scenario_len = 10

        self.scenario_impulses = defaultdict(dict)
        self.impulses = {}
        self.base_values = None
        self.scenario_values = None

        self.process_scenario()

    def reset(self):
        self.graph = self.create_graph()
        self.reset_scenario()

    def reset_scenario(self):
        if self.base_values is not None:
            self.set_values(self.base_values)

        self.impulses = {}
        self.scenario_impulses = defaultdict(dict)

        self.base_values = None
        self.scenario_values = None
        self.process_scenario()

    def read_df(self):
        return pd.read_csv(self.url.format(gid=0), index_col=0).replace('-', 0).fillna(0).rename_axis(
            'Фактор', axis=0).astype(np.float)

    def read_start_values(self):
        return pd.read_csv(self.url.format(gid=592692655), index_col=0).replace('-', 0).fillna(0).rename_axis(
            'Фактор', axis=0).astype(np.float).transpose()

    def get_factor_names(self):
        return self.df.columns

    def get_matrix(self):
        return self.df.to_numpy()

    def create_graph(self):
        graph = nx.DiGraph()
        for name, value in self.start_values['Початкове значення'].items():
            graph.add_node(name, value=value)
        for col_name, column in self.df.items():
            for row_name, weight in column.iteritems():
                if weight != 0:
                    graph.add_edge(row_name, col_name, weight=weight)
        return graph

    def plot_graph_on_figure(self, figure=None):
        if figure is None:
            figure = plt.figure()
        figure.clear()
        figure.set_facecolor('w')
        ax = figure.add_axes((0, 0, 1, 1))
        pos = patches.RegularPolygon(xy=(0.5, 0.5), radius=0.5, numVertices=self.graph.number_of_nodes()).get_verts()
        pos = {node: coord for node, coord in zip(self.graph.nodes.keys(), pos)}
        nx.draw_networkx(
            self.graph, pos=pos, ax=ax,
            with_labels=True,
            font_weight='bold',
            node_shape='H', node_color=[[0.8, 0.3, 0.1]],
            edge_color=[[0.3, 0., 0.9]],
            width=[abs(weight) * 3 for weight in nx.get_edge_attributes(self.graph, 'weight').values()],
            font_size=11, font_family='calibri',
            node_size=[data['value'] * 100 for node, data in self.graph.nodes.items()])
        ax.set_axis_off()
        return figure

    def add_node(self, node, value):
        self.graph.add_node(node, value=value)
        self.df[node] = 0
        self.df.loc[node] = 0

    def add_edge(self, node1, node2, weight):
        self.graph.add_edge(node1, node2, weight=weight)
        self.df.loc[node1, node2] = weight

    def delete_edge(self, node1, node2):
        self.graph.remove_edge(node1, node2)
        self.df.loc[node1, node2] = 0

    def delete_node(self, node):
        self.graph.remove_node(node)
        self.df = self.df.drop(node, axis=0)
        self.df = self.df.drop(node, axis=1)

    def get_values(self):
        return {k: v for k, v in self.graph.nodes(data='value')}

    def set_values(self, values):
        nx.set_node_attributes(self.graph, values, name='value')

    def get_even_cycles(self):
        return [cycle for cycle in nx.simple_cycles(self.graph) if len(cycle) % 2 == 0]

    def add_impulse(self, node, time, value):
        self.scenario_impulses[time][node] = value

    def process_impulse(self, node, value):
        self.graph.nodes[node]['value'] += value

    def time_step(self, step_ind):
        self.impulses.update(self.scenario_impulses[step_ind])
        new_impulses = defaultdict(lambda: 0)
        for node, value in self.impulses.items():
            self.process_impulse(node, value)
            for neighbor in self.graph.neighbors(node):
                new_impulses[neighbor] += value * self.graph.edges[node, neighbor]['weight']
        self.impulses = dict(new_impulses)

    def process_scenario(self):
        self.scenario_values = [self.get_values()]
        for step_ind in range(self.scenario_len):
            self.time_step(step_ind)
            self.scenario_values.append(self.get_values())
        self.set_values(self.scenario_values[0])

    def plot_scenario(self, figure=None):
        if figure is None:
            figure = plt.figure()
        figure.clear()
        legend = []
        for node in self.graph.nodes:
            plt.plot([values[node] for values in self.scenario_values])
            legend.append(node)
        ax = figure.gca()
        ax.legend(legend)
        return figure

    def get_node_dynamics(self, node):
        return [step_values[node] for step_values in self.scenario_values]

    def set_step(self, step):
        self.set_values(self.scenario_values[step])

    def structural_stability(self):
        even_cycles = self.get_even_cycles()
        if even_cycles:
            return 'Система не є структурно стійкою, наявні парні цикли:\n' + '\n'.join(' -> '.join(cycle)
                                                                                       for cycle in even_cycles)
        else:
            return 'Система структурно стійка'

    @staticmethod
    def output_complex(number):
        if number == 0:
            return '0'
        output_string = ''
        if number.real:
            output_string += f'{number.real:.2}'
        if number.imag:
            output_string += ' + ' if number.imag > 0 and number.real else ' - '
            output_string += f'{np.abs(number.imag):.2}i'
        return output_string

    def impulse_stability(self):
        eigen_values = np.linalg.eig(self.get_matrix())[0]
        max_value = np.abs(eigen_values).max()
        output_string = 'Власні числа: ' + ', '.join(self.output_complex(val) for val in set(eigen_values))
        if max_value < 1:
            output_string += '\nСистема стійка за початковим значенням'
        if max_value <= 1:
            output_string += '\nСистема стійка за збуренням'
        else:
            output_string += '\nСистема не є чисельно стійкою'
        return output_string

    def graph_info(self):
        return self.structural_stability() + '\n\n' + self.impulse_stability()
