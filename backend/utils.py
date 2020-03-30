import math

from itertools import chain

import pandas as pd
import numpy as np


def gaussian_density(x, norm_factor, dispersion, mean):
    return (1 / (norm_factor * np.sqrt(2 * math.pi * dispersion))) * np.exp(
        -((x - mean) ** 2) / (2 * dispersion)
    )


def set_argmax(input):
    return np.where(input == np.max(input))[0]


def set_argmin(input):
    return np.where(input == np.min(input))[0]


def parse_number_string(input):
    input = input.replace(",", ".")
    input = input.replace("\n", "\t")
    input = input.split("\t")
    input = list(filter(lambda x: len(x) > 0, input))
    return list(map(float, input))


def create_df_from_string(input):
    number_seq = parse_number_string(input)

    number_table = np.array(
        [number_seq[(i - 1) * 14 : i * 14] for i in range(1, 16 + 1)]
    )

    first_column_name = [
        "Надто низький рівень",
        "Дуже низький рівень",
        "Низький рівень",
        "Середній рівень",
        "Високий рівень",
        "Дуже високий рівень",
        "Надто високий рівень",
    ]
    first_column_name = list(chain(*[[el, el] for el in first_column_name]))
    second_column_name = [
        ["Mu_k_{}".format(i), "V_k_{}".format(i)] for i in range(1, 8)
    ]
    second_column_name = list(chain(*second_column_name))

    df = pd.DataFrame(
        index=["Експерт {}".format(i) for i in range(1, 17)], data=number_table,
    ).reset_index()

    df.columns = [
        np.array([""] + first_column_name),
        np.array([""] + second_column_name),
    ]

    return df
