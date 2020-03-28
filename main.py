from backend import IntegralAwarenessFactors


def run():
    factors = IntegralAwarenessFactors.default_instance()
    factors.plot(2, 2, range(25))
    print(factors.critical_time_range(2, 2))


if __name__ == '__main__':
    run()
