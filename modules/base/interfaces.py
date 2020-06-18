from abc import ABC, abstractmethod


class IDataLoader(ABC):

    @abstractmethod
    def load(self):
        raise NotImplementedError


class ITransformer(ABC):

    @abstractmethod
    def transform(self):
        raise NotImplementedError


class IEstimator(ABC):

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
