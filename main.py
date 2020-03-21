from backend import IntegralAwarenessFactors


def run():
    factors = IntegralAwarenessFactors.default_instance()
    factors.plot(3, 4, range(0, 200))


if __name__ == '__main__':
    run()
