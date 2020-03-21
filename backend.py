import numpy as np
import matplotlib.pyplot as plt


class IntegralAwarenessFactors:
    def __init__(
            self, alphas_0: np.ndarray,
            completeness_0: np.ndarray, credibility_0: np.ndarray, timeliness_0: np.ndarray):
        self.alphas_0 = alphas_0
        self.completeness_0 = completeness_0
        self.credibility_0 = credibility_0
        self.timeliness_0 = timeliness_0

    @classmethod
    def default_instance(cls):
        return cls(**cls.default_values())

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
        return IntegralAwarenessFactors.parse_values_dict(text_dict)

    # @property
    # def alphas(self):
    #     return np.where(self.alphas_0 <= 1, np.exp(self.alphas_0) * self.completeness_0 * 0.5, 0)
    #
    # @property
    # def betas(self):
    #     return np.where(self.alphas_0 <= 1, (self.alphas_0 * self.gammas) * self.timeliness_0 * 1e-5, 0)
    #
    # @property
    # def gammas(self):
    #     return np.where(self.alphas_0 <= 1, np.exp(self.credibility_0) * self.alphas_0 * 0.05, 0)

    @property
    def alphas(self) -> np.ndarray:
        return 1 + (1 + self.alphas_0) * (1 + self.completeness_0 * self.credibility_0 * self.timeliness_0)

    @property
    def betas(self) -> np.ndarray:
        return 1 + (1 + self.alphas_0 * self.credibility_0)

    @property
    def gammas(self) -> np.ndarray:
        return (1 + 0.5 * self.betas * self.alphas_0 ** 2 * self.timeliness_0) ** 2

    def completeness_factor(self, t: int):
        return 0.5 * self.completeness_0 * ((self.alphas + self.gammas) * t) ** 2

    def credibility_factor(self, t: int):
        return 0.5 * self.credibility_0 * ((self.alphas + self.gammas) * t) ** 2

    def timeliness_factor(self, t: int):
        return self.timeliness_0 * (1 + (1 - self.betas * t)) ** 2

    def awareness_factor(self, t: int) -> np.ndarray:
        return np.prod(
            [factor(t) for factor in (self.completeness_factor, self.credibility_factor, self.timeliness_factor)])

    def critical_probability(self, t) -> np.ndarray:
        return 1 - np.log(1 + self.alphas * self.awareness_factor(t))

    def timeseries(self, name: str, i: int, j: int, time_range: range) -> np.ndarray:
        if name == 'completeness':
            factor = self.completeness_factor
        elif name == 'credibility':
            factor = self.credibility_factor
        elif name == 'timeliness':
            factor = self.timeliness_factor
        else:
            raise ValueError(f'timeseries name {name} not recognized')

        return np.array([factor(t)[i, j] for t in time_range])

    def plot(self, i: int, j: int, time_range: range):
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
        axs[0].set_xlabel('time')
        for name, ax, color in zip(('completeness', 'credibility', 'timeliness'), axs, ('r', 'g', 'b')):
            ax.plot(time_range, self.timeseries(name, i, j, time_range), color)
            ax.set_ylabel(f'{name} factor')

        fig.tight_layout()
        plt.show()
