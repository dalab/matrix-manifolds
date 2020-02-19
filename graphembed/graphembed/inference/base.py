import abc


class StochasticGraphModel:

    @abc.abstractmethod
    def __call__(self, theta, ret_margs=False):
        pass
