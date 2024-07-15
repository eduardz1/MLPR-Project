import unittest
from functools import partial

import numpy as np
import scipy.optimize as opt
import sklearn.datasets as datasets

from project.classifiers.logistic_regression import LogisticRegression
from project.funcs.base import split_db_2to1, vcol
from project.funcs.dcf import dcf


class LogRegObjTests(unittest.TestCase):
    def setUp(self):
        def load_iris_binary():
            D, L = datasets.load_iris()["data"].T, datasets.load_iris()["target"]  # type: ignore
            D = D[:, L != 0]  # We remove setosa from D
            L = L[L != 0]  # We remove setosa from L
            L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
            return D, L

        D, L = load_iris_binary()
        (self.DTR, self.LTR), (self.DVAL, self.LVAL) = split_db_2to1(D, L)

    def test_log_reg_obj(self):
        logReg = partial(
            LogisticRegression.objective,
            DTR=self.DTR,
            LTR=self.LTR,
            l=1,
            prior=None,
        )

        x, f, _ = opt.fmin_l_bfgs_b(
            logReg, np.zeros(self.DTR.shape[0] + 1), approx_grad=False
        )

        result = (x, f)

        expected_result = (
            [-0.11040207, -0.02898688, -0.24787109, -0.14950472, 2.31094484],
            0.6316436205354684,
        )

        np.testing.assert_array_almost_equal(result[0], expected_result[0])
        self.assertAlmostEqual(result[1], expected_result[1])

        w, b = x[:-1], x[-1]
        S = vcol(w).T @ self.DVAL + b
        LP = S > 0

        error_rate = np.mean(LP != self.LVAL)

        expected_error_rate = 0.1470588235294118

        np.testing.assert_almost_equal(error_rate, expected_error_rate)

        PRIOR = 0.8
        logReg = partial(
            LogisticRegression.objective,
            DTR=self.DTR,
            LTR=self.LTR,
            l=1e-3,
            prior=PRIOR,
        )

        x, f, _ = opt.fmin_l_bfgs_b(
            logReg, np.zeros(self.DTR.shape[0] + 1), approx_grad=False
        )

        w, b = x[:-1], x[-1]

        S = vcol(w).T @ self.DVAL + b
        LP = S > 0

        error_rate = np.mean(LP != self.LVAL)
        expected_error_rate = 0.11764705882352944

        np.testing.assert_almost_equal(error_rate, expected_error_rate)

        S_llr = S.ravel() - np.log(PRIOR / (1 - PRIOR))

        min_dcf = dcf(S_llr, self.LVAL, PRIOR, "min", normalize=True).item()
        act_dcf = dcf(S_llr, self.LVAL, PRIOR, "optimal", normalize=True).item()

        np.testing.assert_almost_equal(min_dcf, 0.16666666666666666)
        np.testing.assert_almost_equal(act_dcf, 0.2222222222222222)


if __name__ == "__main__":
    unittest.main()
