from backend import IntegralAwareness


def get_outputs(i: int, j: int, time_start: int, time_stop: int, time_step: int, max_prob: float):
    factors = IntegralAwareness.default_instance(max_prob=max_prob, limit_t=time_stop)
    time_range = range(time_start, time_stop, time_step)
    i -= 1
    j -= 1
    return {
        'plot_dict': factors.get_plot_dict(i, j, time_range),
        'critical_time_range': factors.critical_time_range(i),
        'classification_df': factors.classification_df()
    }


def run():
    print(get_outputs(3, 3, 0, 70, 1, 0.7))


if __name__ == '__main__':
    run()
