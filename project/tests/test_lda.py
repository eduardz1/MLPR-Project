import os
import unittest

import numpy as np
import sklearn.datasets as skdata

from project.funcs.lda import lda


class LDATest(unittest.TestCase):

    def setUp(self):
        self.IRIS_LDA_matrix_m2 = np.load(
            os.path.join(os.path.dirname(__file__), "data/lda/iris_lda_matrix_m2.npy")
        )

        self.D, self.L = skdata.load_iris()["data"], skdata.load_iris()["target"]  # type: ignore

    def test_lda(self):
        eigvec, _ = lda(self.D, self.L, 2)

        np.testing.assert_allclose(eigvec, self.IRIS_LDA_matrix_m2, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
