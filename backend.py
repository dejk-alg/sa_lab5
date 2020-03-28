from typing import Callable, Optional, Iterable
from collections import defaultdict
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame


def lazy_wrapper(func: Callable):
    def lazy_func(self, t: int):
        if t not in self.cached_values[func].keys():
            self.cached_values[func][t] = func(self, t)
        return self.cached_values[func][t]
    return lazy_func


class IntegralAwareness:
    def __init__(
            self, alphas_0: np.ndarray,
            completeness_0: np.ndarray, credibility_0: np.ndarray, timeliness_0: np.ndarray,
            max_prob: float, limit_t: Optional[int] = None):
        self.alphas_0 = alphas_0
        self.completeness_0 = completeness_0
        self.credibility_0 = credibility_0
        self.timeliness_0 = timeliness_0
        self.cached_values = defaultdict(dict)
        self.alphas = self.betas = self.gammas = None
        self.situations_amount = self.factors_amount = None
        self.set_coeffs()
        self.max_prob = max_prob
        self.limit_t = limit_t

    def valid_factors(self, i):
        return np.where(self.alphas_0[i] != -1)

    @classmethod
    def default_instance(cls, max_prob: float, limit_t: Optional[int] = None):
        return cls(**cls.default_values(), max_prob=max_prob, limit_t=limit_t)

    @staticmethod
    def parse_values_dict(string_dict):
        return {
            name: np.stack([np.fromstring(line, dtype=float, sep=' ')
                            for line in inp_string.replace('-', '-1.').split('\n') if line.strip()], axis=0)
            for name, inp_string in string_dict.items()}

    @staticmethod
    def default_values():
        text_dict = {
            'alphas_0':
                '''
                0.6 0.5 0.4 0.55 - 0.755 0.45
                - - 0.7 0.35 - - 0.7
                0.65 - 0.8 0.65 0.7 0.65 0.7
                - - 0.4 0.55 0.45 0.85 -
                ''',
            'completeness_0':
                '''
                0.65 0.55 0.8 0.45 - 0.7 0.75
                - - 0.5 0.8 - - 0.45
                0.45 - 0.6 0.5 0.6 0.45 0.45
                - - 0.7 0.7 0.4 0.35 -
                ''',
            'credibility_0':
                '''
                0.3 0.54 0.5 0.4 - 0.5 0.4
                - - 0.65 0.25 - - 0.5
                0.35 - 0.35 0.4 0.2 0.45 0.3
                - - 0.65 0.5 0.65 0.3 -
                ''',
            'timeliness_0':
                '''
                0.7 0.8 0.4 0.6 - 0.85 0.5
                - - 0.5 0.4 - - 0.4
                0.25 - 0.76 0.45 0.3 0.5 0.6
                - - 0.5 0.5 0.6 0.3 -
                '''

        }
        return IntegralAwareness.parse_values_dict(text_dict)

    def set_coeffs(self):
        self.alphas = 1 + (1 + self.alphas_0) * (1 + self.completeness_0 * self.credibility_0 * self.timeliness_0)
        self.betas = 1 + (1 + self.alphas_0 * self.credibility_0)
        self.gammas = (1 + 0.5 * self.betas * self.alphas_0 ** 2 * self.timeliness_0) ** 2
        self.situations_amount, self.factors_amount = self.alphas.shape

    def completeness_factor(self, t: int):
        return 1 - (1 - self.completeness_0) * np.exp(-1e-4 * ((self.alphas + self.gammas) * t) ** 2)

    def credibility_factor(self, t: int):
        return 1 - (1 - self.credibility_0) * np.exp(-1e-4 * ((self.alphas + self.gammas) * t) ** 2)

    def timeliness_factor(self, t: int):
        return self.timeliness_0 * np.exp(-1e-2 * self.betas * t)

    def awareness_factor(self, t: int) -> np.ndarray:
        return np.prod(
            [fact(t) for fact in (self.completeness_factor, self.credibility_factor, self.timeliness_factor)], axis=0)

    def critical_probability(self, t: int) -> np.ndarray:
        return 1 - np.log(1 + self.alphas * self.awareness_factor(t))

    for attr_name in (
            'completeness_factor', 'credibility_factor', 'timeliness_factor', 'awareness_factor',
            'critical_probability'):
        locals()[attr_name] = lazy_wrapper(locals()[attr_name])

    def timeseries(self, name: str, i: int, j: int, time_range: Iterable) -> np.ndarray:
        if name in ('completeness', 'credibility', 'timeliness', 'awareness'):
            factor = getattr(self, f'{name}_factor')
        elif name == 'critical_probability':
            factor = getattr(self, name)
        else:
            raise ValueError(f'timeseries name {name} not recognized')

        return np.array([factor(t)[i, j] for t in time_range])

    def find_best_prob(self, i: int, j: int):
        prev_prob = 1
        for t in count():
            crit_prob_unscaled = self.critical_probability(t)[i, j]
            if crit_prob_unscaled > prev_prob:
                return t - 1, prev_prob
            prev_prob = crit_prob_unscaled
            if self.limit_t is not None and t > self.limit_t:
                return None

    def critical_time_range_per_factor(self, i: int, j: int):
        best_t, best_prob = self.find_best_prob(i, j)
        scaled_max_prob = self.max_prob * (1 - best_prob) + best_prob
        min_t = None
        for t in count():
            if min_t is None and self.critical_probability(t)[i, j] < scaled_max_prob:
                min_t = t
            if min_t is not None and self.critical_probability(t)[i, j] > scaled_max_prob:
                max_t = t - 1
                return min_t, max_t
            if self.limit_t is not None and t >= self.limit_t:
                if min_t is None:
                    raise ValueError
                max_t = self.limit_t
                return min_t, max_t

    def critical_time_range(self, i: int):
        time_ranges = [self.critical_time_range_per_factor(i, j,) for j in self.valid_factors(i)]
        return max(map(lambda x: x[0], time_ranges)), min(map(lambda x: x[1], time_ranges))

    def classify_situation(self, i: int):
        time_range = self.critical_time_range(i)
        time = 20
        if time <= time_range[0]:
            return 'A1 - особливо небезпечна ситуація'
        elif time <= time_range[1]:
            return 'A2 - потенційно небезпечна ситуація'
        else:
            return 'A3 - майже безпечна ситуація'

    def classification_df(self):
        classification = {i: self.classify_situation(i) for i in range(self.situations_amount)}
        classification = DataFrame(classification)
        return classification

    def create_timeseries_df(self, i: int, j: int, time_range: Iterable):
        timeseries = {name: self.timeseries(name, i, j, time_range)
                      for name in ('completeness', 'credibility', 'timeliness', 'awareness')}
        timeseries.update({'time': list(time_range)})
        timeseries['values'] = {'critical_probability': self.timeseries('critical_probability', i, j, time_range)}
        timeseries = DataFrame(timeseries)
        return timeseries

    def get_plot_dict(self, i: int, j: int, time_range: Iterable):
        plot_dict = {f'{name}_factor': self.timeseries(name, i, j, time_range)
                     for name in ('completeness', 'credibility', 'timeliness', 'awareness')}
        plot_dict['time'] = list(time_range)
        return plot_dict

    def plot(self, i: int, j: int, time_range: Iterable):
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
        for ax in axs[:, -1]:
            ax.set_xlabel('time')
        for name, ax, color in zip(
                ('completeness', 'credibility', 'timeliness', 'awareness'), axs.flatten(), ('r', 'g', 'b', 'y')):
            ax.plot(time_range, self.timeseries(name, i, j, time_range), color)
            ax.set_ylabel(f'{name} factor')
            ax.set_ylim((-0.1, 1.1))

        fig.tight_layout()
        plt.show()
        plt.close(fig)
