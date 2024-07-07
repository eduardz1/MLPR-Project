import json
import os
import unittest

import numpy as np

from project.funcs.log_pdf import log_pdf_gmm


class LogPdfTests(unittest.TestCase):
    def test_log_pdf_gmm(self):
        def load_gmm(filename):
            with open(filename, "r") as f:
                gmm = json.load(f)
            return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]

        GMM_data_4D = np.load(
            os.path.join(os.path.dirname(__file__), "data/log_pdf/GMM_data_4D.npy")
        )
        GMM_4D_3G_init_ll = np.load(
            os.path.join(
                os.path.dirname(__file__), "data/log_pdf/GMM_4D_3G_init_ll.npy"
            )
        )
        GMM_4D_3G_init = load_gmm(
            os.path.join(os.path.dirname(__file__), "data/log_pdf/GMM_4D_3G_init.json")
        )

        np.testing.assert_almost_equal(
            # For some reason the output is a 2D array with a single row
            np.expand_dims(log_pdf_gmm(GMM_data_4D, GMM_4D_3G_init), axis=0),
            GMM_4D_3G_init_ll,
        )

        GMM_data_1D = np.load(
            os.path.join(os.path.dirname(__file__), "data/log_pdf/GMM_data_1D.npy")
        )

        GMM_1D_3G_init_ll = np.load(
            os.path.join(
                os.path.dirname(__file__), "data/log_pdf/GMM_1D_3G_init_ll.npy"
            )
        )
        GMM_1D_3G_init = load_gmm(
            os.path.join(os.path.dirname(__file__), "data/log_pdf/GMM_1D_3G_init.json")
        )

        np.testing.assert_almost_equal(
            # For some reason the output is a 2D array with a single row
            np.expand_dims(log_pdf_gmm(GMM_data_1D, GMM_1D_3G_init), axis=0),
            GMM_1D_3G_init_ll,
        )


if __name__ == "__main__":
    unittest.main()
