from modules.base.interfaces import IEstimator


class Pipeline(IEstimator):

    def __init__(self, steps: list):
        self.steps = steps

    def fit(self, X: np.array, y: np.array) -> self:
        raise NotImplementedError

    def predict(self, X: np.array) -> np.array:
        raise NotImplementedError

    def save(self, path: str) -> self:
        raise NotImplementedError

    def load(self, path: str) -> self:
        raise NotImplementedError
