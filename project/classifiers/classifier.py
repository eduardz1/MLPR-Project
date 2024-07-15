import abc
from io import TextIOWrapper

import numpy as np
import numpy.typing as npt


class Classifier(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, X: npt.NDArray[np.float64], y: npt.ArrayLike, *args) -> "Classifier":
        pass

    @abc.abstractmethod
    def scores(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Compute the scores of the classifier.

        Args:
            X (npt.NDArray[np.float64]): The validation set.

        Raises:
            ValueError: Classifier has not been fitted yet.

        Returns:
            npt.NDArray[np.float64]: Scores of the classifier.
        """
        pass

    @property
    @abc.abstractmethod
    def llr(self) -> npt.NDArray[np.float64]:
        pass

    @abc.abstractmethod
    def to_json(self, fp=None) -> None | dict:
        """
        Serialize the classifier to a JSON file.

        Args:
            fp: The file pointer to write the JSON data to, if None, return the
                JSON data as a dictionary.

        Raises:
            ValueError: Classifier has not been fitted yet.

        Returns:
            None | dict: The JSON data of the classifier.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def from_json(data: str | bytes | bytearray | TextIOWrapper) -> "Classifier":
        """
        Load a classifier from a JSON string.

        Args:
            data (str | bytes | bytearray): JSON string containing the classifier data.

        Returns:
            Classifier: The loaded classifier.
        """
        pass
