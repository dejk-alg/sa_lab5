import warnings

from itertools import chain
from runpy import run_path

import pandas as pd
import numpy as np
import scipy.optimize as optimize
import seaborn as sns

from scipy.special import softmax
from sklearn.metrics import auc
from matplotlib import pyplot as plt

from .utils import gaussian_density, set_argmax, set_argmin, create_df_from_string


class DelfiMethod(object):
    def __init__(
        self,
        cases,
        criteria,
        K,
        S_star,
        R_interval,
        experts_proof,
        W,
        tables,
        xs,
        levels,
    ):

        self.cases = cases
        self.criteria = criteria
        self.K = K
        self.S_star = S_star
        self.R_interval = R_interval
        self.experts_proof = experts_proof
        self.W = W
        self.tables = tables
        self.xs = xs
        self.solutions = levels

        self.n_experts = len(self.experts_proof)
        self.n_solutions = len(self.solutions)
        self.n_criteria = len(self.criteria)
        self.n_cases = len(self.cases)

        self.experts_df = {
            c: {
                v: create_df_from_string(self.tables[c][v])
                for v in self.tables[c].keys()
            }
            for c in self.tables.keys()
        }
        self.experts_intervals = {
            c: {
                v: self.compute_df_interval(self.experts_df[c][v])
                for v in self.experts_df[c].keys()
            }
            for c in self.experts_df.keys()
        }
        self.expected_values = {
            c: {
                v: self.compute_expected_value(self.experts_intervals[c][v])
                for v in self.experts_df[c].keys()
            }
            for c in self.experts_df.keys()
        }
        self.integral_values = {
            c: {
                v: self.compute_integral_expert_mark(
                    self.experts_intervals[c][v], self.expected_values[c][v]
                )
                for v in self.experts_df[c].keys()
            }
            for c in self.experts_df.keys()
        }
        self.integral_gaussian_values = {
            c: {
                v: self.compute_integral_gaussian_mark(self.integral_values[c][v])
                for v in self.experts_df[c].keys()
            }
            for c in self.experts_df.keys()
        }
        self.expert_conf = {
            c: {
                v: self.compute_quality_functional(
                    self.experts_df[c][v], self.integral_gaussian_values[c][v]
                )
                for v in self.experts_df[c].keys()
            }
            for c in self.experts_df.keys()
        }
        self.expert_distance_matrices = {
            c: {
                v: self.compute_expert_distance_matrix(self.experts_intervals[c][v])
                for v in self.experts_df[c].keys()
            }
            for c in self.experts_df.keys()
        }
        self.median_expert = {
            c: {
                v: self.compute_median_expert(self.expert_distance_matrices[c][v])
                for v in self.experts_df[c].keys()
            }
            for c in self.experts_df.keys()
        }
        self.experts_belive_interval = {
            c: {
                v: self.compute_believe_intervals(
                    self.median_expert[c][v],
                    self.expert_distance_matrices[c][v],
                    self.expert_conf[c][v],
                )
                for v in self.experts_df[c].keys()
            }
            for c in self.experts_df.keys()
        }

        self.best_solution = {
            c: {
                v: self.compute_best_mark(
                    self.median_expert[c][v], self.experts_intervals[c][v]
                )
                for v in self.experts_df[c].keys()
            }
            for c in self.experts_df.keys()
        }

        self.overall_info_df = self.compute_overall_df()
        self.mark_df_numerical = self.compute_mark_df_numerical()
        self.mark_df_values = self.compute_mark_df_in_values()

    @staticmethod
    def initialize_from_py_file(path):
        settings = run_path(path)
        return DelfiMethod(
            cases=settings["cases"],
            criteria=settings["criteria"],
            K=settings["K"],
            S_star=settings["S_star"],
            R_interval=settings["R_interval"],
            experts_proof=settings["experts_proof"],
            W=settings["W"],
            tables=settings["tables"],
            xs=settings["xs"],
            levels=settings["levels"],
        )

    @staticmethod
    def extract_mu_values(df):
        return df.loc[
            :,
            list(
                filter(
                    lambda x: x[1].startswith("Mu")
                    if isinstance(x, tuple)
                    else x.startswith("Mu"),
                    df.columns,
                )
            ),
        ]

    @staticmethod
    def extract_v_values(df):
        return df.loc[
            :,
            list(
                filter(
                    lambda x: x[1].startswith("V")
                    if isinstance(x, tuple)
                    else x.startswith("V"),
                    df.columns,
                )
            ),
        ]

    @staticmethod
    def extract_dmin_values(df):
        return df.loc[
            :,
            list(
                filter(
                    lambda x: x[1].startswith("dmin")
                    if isinstance(x, tuple)
                    else x.startswith("dmin"),
                    df.columns,
                )
            ),
        ]

    @staticmethod
    def extract_dmax_values(df):
        return df.loc[
            :,
            list(
                filter(
                    lambda x: x[1].startswith("dmax")
                    if isinstance(x, tuple)
                    else x.startswith("dmax"),
                    df.columns,
                )
            ),
        ]

    @staticmethod
    def extract_expert_values(df, expert_id):
        expert_raw = df.iloc[expert_id - 1, :]
        expert_raw = [
            [expert_raw[i], expert_raw[i + 1], expert_raw[i + 2]]
            for i in range(0, len(expert_raw), 3)
        ]
        return np.array(expert_raw).T

    @staticmethod
    def compute_expert_distances(exp_id, d_min, d_max):
        dist = np.stack(
            [np.abs(d_min - d_min[exp_id]), np.abs(d_max - d_max[exp_id])], axis=-1
        ).max(axis=-1)

        dist = dist.mean(axis=1)
        return dist

    def compute_interval_points(self, mu, v):
        d_min = np.abs(mu - mu * (1 - v) * self.K)
        d_min[d_min < 0] = 0
        d_max = np.abs(mu + mu * (1 - v) * self.K)
        d_max[d_max > 1] = 1
        return d_min, d_max

    def level_to_number(self, level):
        return self.xs[self.solutions.index(level)]

    def color_value_df(self, input):
        return [
            "background-color: rgba(0, 255, 0, {})".format(self.level_to_number(el))
            for el in input
        ]

    def compute_mark_df_in_values(self):
        mark_df = []

        for case in self.cases:
            mark_df.append(
                [
                    self.solutions[self.best_solution[case][criteria][0]]
                    for criteria in self.criteria
                ]
            )

        mark_df = pd.DataFrame(mark_df)
        mark_df.index = self.cases
        mark_df.columns = self.criteria

        mark_df = mark_df.style.apply(self.color_value_df)

        return mark_df

    def compute_mark_df_numerical(self):
        mark_df = []

        for case in self.cases:
            mark_df.append(
                [
                    self.xs[self.best_solution[case][criteria][0]]
                    for criteria in self.criteria
                ]
            )

        mark_df = np.array(mark_df)
        case_marks = (mark_df * self.W).mean(axis=1)
        case_lb = np.argsort(-case_marks) + 1

        mark_df = np.concatenate(
            [np.expand_dims(case_lb, -1), mark_df, np.expand_dims(case_marks, -1)],
            axis=-1,
        )

        mark_df = pd.DataFrame(mark_df)
        mark_df.index = self.cases
        mark_df.columns = [
            np.array(["№"] + self.criteria + ["Wn"]),
            np.array([""] + self.W + [""]),
        ]

        cm = sns.light_palette("green", as_cmap=True)
        mark_df = mark_df.style.background_gradient(
            cmap=cm, subset=mark_df.columns[1:-1]
        )

        return mark_df

    def compute_overall_df(self):
        overall_info_df = []
        for case in self.cases:

            # Number of accepted experts
            overall_info_df.append(
                [
                    self.experts_belive_interval[case][criteria].sum()
                    for criteria in self.criteria
                ]
            )

            # Percantage of accepted experts
            percantage = []
            for criteria in self.criteria:
                criteria_believe = self.experts_belive_interval[case][criteria]
                percantage.append(
                    (criteria_believe.sum() / criteria_believe.shape[0]) * 100
                )
            overall_info_df.append(percantage)

            # Median expert
            overall_info_df.append(
                [
                    "Expert {}".format(self.median_expert[case][criteria] + 1)
                    for criteria in self.criteria
                ]
            )

            # Get best mark mu and approval
            mu, level = [], []
            for criteria in self.criteria:
                info = self.best_solution[case][criteria]
                level.append(info[0] + 1)
                mu.append(info[1])

            overall_info_df.append(np.array(percantage) / 100)
            overall_info_df.append(mu)
            overall_info_df.append(level)

        overall_info_df = pd.DataFrame(overall_info_df)

        overall_info_df.index = [
            np.array(list(chain(*[[el] * 6 for el in self.cases]))),
            np.array(["exp", "%", "Mid", "S", "Q", "s"] * len(self.cases)),
        ]

        overall_info_df.columns = self.criteria

        return overall_info_df

    def compute_best_mark(self, median_expert, df_intervals):
        d_min = DelfiMethod.extract_dmin_values(df_intervals).values
        d_max = DelfiMethod.extract_dmax_values(df_intervals).values
        mu = DelfiMethod.extract_mu_values(df_intervals).values

        d_min = d_min[median_expert, :]
        d_max = d_max[median_expert, :]
        mu = mu[median_expert, :]

        best_solution = set_argmax(mu)

        if len(best_solution) > 1:
            most_confident_solution = set_argmin(np.abs(d_min - d_max))
            if set(best_solution) & set(most_confident_solution):
                best_solution = list(set(best_solution) & set(most_confident_solution))[
                    0
                ]
            else:
                best_solution = best_solution[0]

        else:
            best_solution = best_solution[0]

        return (
            best_solution,
            mu[best_solution],
            1 - abs(d_max[best_solution] - d_min[best_solution]),
        )

    def compute_believe_intervals(
        self, median_expert, epert_distance_matrix, expert_conf
    ):
        dist = epert_distance_matrix[median_expert]

        believe_interval = dist * (2 - expert_conf)

        return believe_interval < self.R_interval

    def compute_median_expert(self, epert_distance_matrix):
        return np.argmin(epert_distance_matrix.sum(axis=1))

    def compute_expert_distance_matrix(self, df_intervals):
        d_min = DelfiMethod.extract_dmin_values(df_intervals).values
        d_max = DelfiMethod.extract_dmax_values(df_intervals).values

        experts_distance_matrix = []
        for exp_id in range(d_min.shape[0]):
            dist = DelfiMethod.compute_expert_distances(exp_id, d_min, d_max)
            experts_distance_matrix.append(dist)

        return np.array(experts_distance_matrix)

    def compute_quality_functional(self, df, df_integral_gaussian):
        mu = DelfiMethod.extract_mu_values(df).values
        mu_min = np.abs(
            mu
            - np.expand_dims(df_integral_gaussian["model_gaussian_down"].values, axis=0)
        )
        mu_max = np.abs(
            mu
            - np.expand_dims(df_integral_gaussian["model_gaussian_up"].values, axis=0)
        )

        mu = np.stack([mu_min, mu_max], axis=-1)
        mu = np.max(mu, axis=-1)
        dist = mu.mean(axis=1)

        return (1 - dist) * np.array(self.experts_proof)

    def compute_integral_gaussian_mark(self, df_integral):

        k_min = 1 / auc(self.xs, df_integral["model_down"])
        k_max = 1 / auc(self.xs, df_integral["model_up"])

        xs = np.array(self.xs)

        def func_to_optimize(params):
            mu, d = params
            return np.abs(
                gaussian_density(xs, k_min, d, mu) - df_integral["model_down"]
            ).sum()

        result_min = optimize.minimize(
            func_to_optimize, [1, 1], bounds=[(-10000, 10000), (1e-12, 10000)]
        )

        if not result_min.success:
            warnings.warn("did not converge", DeprecationWarning)

        def func_to_optimize(params):
            mu, d = params
            return np.abs(
                gaussian_density(xs, k_max, d, mu) - df_integral["model_up"]
            ).sum()

        result_max = optimize.minimize(
            func_to_optimize, [1, 1], bounds=[(-10000, 10000), (1e-12, 10000)]
        )

        if not result_max.success:
            warnings.warn("did not converge", DeprecationWarning)

        qmin_gaussian = gaussian_density(xs, k_min, result_min.x[1], result_min.x[0])
        qmax_gaussian = gaussian_density(xs, k_max, result_max.x[1], result_max.x[0])

        return pd.DataFrame(
            data=np.array([qmin_gaussian, qmax_gaussian]).T,
            columns=["model_gaussian_down", "model_gaussian_up"],
        )

    def compute_integral_expert_mark(self, df_marks, df_expectations):
        mu_values = DelfiMethod.extract_mu_values(df_marks).values
        m_exp_min = df_expectations["dmin_expected"].values
        m_exp_max = df_expectations["dmax_expected"].values

        q_min = np.argmin(np.abs(mu_values - m_exp_min), axis=0)
        q_max = np.argmin(np.abs(mu_values - m_exp_max), axis=0)

        q_min = [mu_values[q_min[i], i] for i in range(q_min.shape[0])]
        q_max = [mu_values[q_max[i], i] for i in range(q_max.shape[0])]

        return pd.DataFrame(
            data=np.array([q_min, q_max]).T, columns=["model_down", "model_up"]
        )

    def compute_df_interval(self, df):
        mu_values = DelfiMethod.extract_mu_values(df).values
        v_values = DelfiMethod.extract_v_values(df).values

        mu_values = mu_values.T
        v_values = v_values.T

        df_interval = []
        for mu, v in zip(mu_values, v_values):
            d_min, d_max = self.compute_interval_points(mu, v)
            df_interval += [d_min, mu, d_max]

        np.array(df_interval).T
        df_interval = pd.DataFrame(
            data=np.array(df_interval).T,
            columns=list(
                chain(
                    *[
                        [
                            "dmin_k_{}".format(i),
                            "Mu_k_{}".format(i),
                            "dmax_k_{}".format(i),
                        ]
                        for i in range(1, self.n_solutions + 1)
                    ]
                )
            ),
        )

        return df_interval

    def compute_expected_value(self, df):
        dmin_values = DelfiMethod.extract_dmin_values(df).values
        dmax_values = DelfiMethod.extract_dmax_values(df).values
        mu_values = DelfiMethod.extract_mu_values(df).values

        expected_df = [
            dmin_values.mean(axis=0),
            mu_values.mean(axis=0),
            dmax_values.mean(axis=0),
        ]

        expected_df = pd.DataFrame(
            data=np.array(expected_df).T,
            columns=["dmin_expected", "mu_expected", "dmax_expected"],
        )

        return expected_df

    def plot_point_predictions(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values
        for i in range(self.n_experts):
            plt.plot(mu_values[i], "-o", label="Expert {}".format(i + 1))

        plt.title("Point estimate")
        plt.legend()
        plt.show()

    def plot_interval_predictions(self, case, crit, expert_id):
        df = self.experts_intervals[case][crit]
        interval_pred = DelfiMethod.extract_expert_values(df, expert_id)
        plt.plot(self.xs, interval_pred[0], "-^", label="low mean")
        plt.plot(self.xs, interval_pred[1], "-o", label="mean")
        plt.plot(self.xs, interval_pred[2], "-v", label="high mean")

        plt.title("Interval estimate of {} expert".format(expert_id))
        plt.legend()
        plt.show()

    def plot_expected_predictions(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values
        for i in range(self.n_experts):
            plt.plot(
                self.xs,
                mu_values[i],
                label="Expert {}".format(i + 1),
                color="tan",
                linewidth=1,
            )

        df_expected = self.expected_values[case][crit]

        plt.plot(
            self.xs,
            df_expected["dmin_expected"],
            "^-",
            label="low mean",
            linewidth=3.0,
            color="green",
        )
        plt.plot(
            self.xs,
            df_expected["mu_expected"],
            "o-",
            label="mean",
            linewidth=3.0,
            color="red",
        )
        plt.plot(
            self.xs,
            df_expected["dmax_expected"],
            "v-",
            label="high mean",
            linewidth=3.0,
            color="green",
        )

        plt.title("Interval estimate of the mean of interval estimates")
        plt.legend()
        plt.show()

    def plot_integral_predictions(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values
        for i in range(self.n_experts):
            plt.plot(
                self.xs,
                mu_values[i],
                label="Expert {}".format(i + 1),
                color="tan",
                linewidth=1,
            )

        df_expected = self.expected_values[case][crit]

        plt.plot(
            self.xs,
            df_expected["dmin_expected"],
            "*-",
            label="low mean",
            linewidth=2.0,
            color="red",
        )
        plt.plot(
            self.xs,
            df_expected["mu_expected"],
            "o-",
            label="mean",
            linewidth=2.0,
            color="maroon",
        )
        plt.plot(
            self.xs,
            df_expected["dmax_expected"],
            "o-",
            label="high mean",
            linewidth=2.0,
            color="red",
        )

        df_integral = self.integral_values[case][crit]

        plt.plot(
            self.xs, df_integral["model_down"], "^-", label="Model -", linewidth=3.0
        )
        plt.plot(self.xs, df_integral["model_up"], "v-", label="Model +", linewidth=3.0)

        plt.title("Interval integrated expert estimate")
        plt.legend()
        plt.show()

    def plot_integral_gaussian(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values

        for i in range(self.n_experts):
            plt.plot(
                self.xs,
                mu_values[i],
                label="Expert {}".format(i + 1),
                color="tan",
                linewidth=1,
            )

        df_integral = self.integral_values[case][crit]

        plt.plot(
            self.xs,
            df_integral["model_down"],
            "*-",
            label="Model +",
            linewidth=2.0,
            color="red",
        )
        plt.plot(
            self.xs,
            df_integral["model_up"],
            "o-",
            label="Model -",
            linewidth=2.0,
            color="red",
        )

        df_integral_gaussian = self.integral_gaussian_values[case][crit]

        plt.plot(
            self.xs,
            df_integral_gaussian["model_gaussian_up"],
            "v-",
            label="Gauss +",
            linewidth=3.0,
            color="green",
        )
        plt.plot(
            self.xs,
            df_integral_gaussian["model_gaussian_down"],
            "^-",
            label="Gauss -",
            linewidth=3.0,
            color="green",
        )

        plt.title("Discrete interval gaussian density")
        plt.legend()
        plt.show()

    def plot_quality_functional(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values

        expert_values = self.expert_conf[case][crit]
        expert_values = (expert_values - expert_values.min()) / (
            expert_values.max() - expert_values.min()
        )
        for i in range(self.n_experts):
            plt.plot(
                self.xs,
                mu_values[i],
                "o-",
                label="Expert {}".format(i + 1),
                linewidth=1.0,
                color=(1.0, 0.0, 0.0, expert_values[i]),
            )

        df_integral_gaussian = self.integral_gaussian_values[case][crit]

        plt.plot(
            self.xs,
            df_integral_gaussian["model_gaussian_up"],
            "v--",
            label="Gauss +",
            linewidth=2.0,
            color="green"
        )
        plt.plot(
            self.xs,
            df_integral_gaussian["model_gaussian_down"],
            "^--",
            label="Gauss -",
            linewidth=2.0,
            color="green"
        )

        plt.title("Estimate of experts by the lowest and the highest quality value")
        plt.legend()
        plt.show()

    def plot_expert_distance_heatmap(self, case, crit):
        sns.heatmap(
            self.expert_distance_matrices[case][crit], annot=True, linewidths=0.5
        )

        plt.title("""Heatmap of experts' estimates distances""")
        plt.show()

    def plot_median_expert(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values

        median_expert = self.median_expert[case][crit]

        df_median = self.experts_intervals[case][crit]
        interval_pred = DelfiMethod.extract_expert_values(df_median, median_expert + 1)

        for i in range(self.n_experts):
            if i == median_expert:
                plt.plot(self.xs, interval_pred[0], "-o", label="d_min", linewidth=2.0)
                plt.plot(
                    self.xs,
                    interval_pred[1],
                    "-o",
                    label="Expert median {}".format(i + 1),
                    linewidth=3.0,
                )
                plt.plot(self.xs, interval_pred[2], "-o", label="d_max", linewidth=2.0)
            else:
                plt.plot(
                    self.xs,
                    mu_values[i],
                    "-o",
                    label="Expert {}".format(i + 1),
                    linewidth=0.5,
                )

        plt.title("Median interval estimate")
        plt.legend()
        plt.show()

    def plot_believed_experts(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values

        median_expert = self.median_expert[case][crit]
        best_solution = self.best_solution[case][crit]

        df_median = self.experts_intervals[case][crit]
        interval_pred = DelfiMethod.extract_expert_values(df_median, median_expert + 1)

        believed_intervals = self.experts_belive_interval[case][crit]
        believed_experts = np.where(believed_intervals)[0]

        cluster_confidence = believed_intervals.sum() / believed_intervals.shape[0]

        print(
            "Estimates in cluster are agreed"
            if cluster_confidence > self.S_star
            else "Estimates in cluster are not agreed"
        )
        print(
            "Best choice is: {}.\nWith the probability: {}.\nAnd confidence: {}".format(
                self.solutions[best_solution[0]],
                best_solution[1],
                cluster_confidence * 100,
            )
        )

        for i in range(self.n_experts):
            if i == median_expert:
                plt.plot(
                    self.xs,
                    interval_pred[0],
                    "^-",
                    label="low mean",
                    linewidth=2.0,
                    color="blue",
                )
                plt.plot(
                    self.xs,
                    interval_pred[1],
                    "o-",
                    label="Expert median {}".format(i + 1),
                    linewidth=3.0,
                    color="cyan",
                )
                plt.plot(
                    self.xs,
                    interval_pred[2],
                    "v-",
                    label="high mean",
                    linewidth=2.0,
                    color="blue",
                )
            elif i in believed_experts:
                plt.plot(
                    self.xs,
                    mu_values[i],
                    "-o",
                    label="Expert {}".format(i + 1),
                    linewidth=1.0,
                    color="green",
                )
            else:
                plt.plot(
                    self.xs,
                    mu_values[i],
                    "-o",
                    label="Expert {}".format(i + 1),
                    linewidth=1.0,
                    color="red",
                )

        plt.title("""Experts' estimates that are/aren't in confidence interval""")
        plt.legend()
        plt.show()
