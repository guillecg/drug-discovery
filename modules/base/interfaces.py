from abc import ABCMeta, abstractmethod


class IDataLoader(metaclass=ABCMeta):

    @abstractmethod
    def load(self):
        raise NotImplementedError


class ITransformer(metaclass=ABCMeta):

    @abstractmethod
    def transform(self):
        raise NotImplementedError


class IEstimator(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError
