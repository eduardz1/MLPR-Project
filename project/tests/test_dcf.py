import os
import unittest

import numpy as np

from project.funcs.dcf import dcf


class DCFTest(unittest.TestCase):

    def setUp(self):
        self.commedia_labels_infpar = np.load(
            os.path.join(
                os.path.dirname(__file__), "data/dcf/commedia_labels_infpar.npy"
            )
        )

        self.commedia_llr_infpar = np.load(
            os.path.join(os.path.dirname(__file__), "data/dcf/commedia_llr_infpar.npy")
        )

    def test_optimal_non_normalized_dcf(self):
        dcfs = []

        dcfs.append(
            dcf(
                llr=self.commedia_llr_infpar,
                y_val=self.commedia_labels_infpar,
                pi=0.5,
                Cf_n=1,
                Cf_p=1,
                strategy="optimal",
                normalize=False,
            )
        )

        dcfs.append(
            dcf(
                llr=self.commedia_llr_infpar,
                y_val=self.commedia_labels_infpar,
                pi=0.8,
                Cf_n=1,
                Cf_p=1,
                strategy="optimal",
                normalize=False,
            )
        )

        dcfs.append(
            dcf(
                llr=self.commedia_llr_infpar,
                y_val=self.commedia_labels_infpar,
                pi=0.5,
                Cf_n=10,
                Cf_p=1,
                strategy="optimal",
                normalize=False,
            )
        )

        dcfs.append(
            dcf(
                llr=self.commedia_llr_infpar,
                y_val=self.commedia_labels_infpar,
                pi=0.8,
                Cf_n=1,
                Cf_p=10,
                strategy="optimal",
                normalize=False,
            )
        )

        expected = [
            0.25557213930348255,
            0.22517412935323383,
            1.1178482587064678,
            0.7235124378109452,
        ]

        np.testing.assert_allclose(dcfs, expected, rtol=1e-5)

    def test_normalized_optimal_dcf(self):
        dcfs = []

        dcfs.append(
            dcf(
                llr=self.commedia_llr_infpar,
                y_val=self.commedia_labels_infpar,
                pi=0.5,
                Cf_n=1,
                Cf_p=1,
                strategy="optimal",
            )
        )

        dcfs.append(
            dcf(
                llr=self.commedia_llr_infpar,
                y_val=self.commedia_labels_infpar,
                pi=0.8,
                Cf_n=1,
                Cf_p=1,
                strategy="optimal",
            )
        )

        dcfs.append(
            dcf(
                llr=self.commedia_llr_infpar,
                y_val=self.commedia_labels_infpar,
                pi=0.5,
                Cf_n=10,
                Cf_p=1,
                strategy="optimal",
            )
        )

        dcfs.append(
            dcf(
                llr=self.commedia_llr_infpar,
                y_val=self.commedia_labels_infpar,
                pi=0.8,
                Cf_n=1,
                Cf_p=10,
                strategy="optimal",
            )
        )

        expected = [
            0.5111442786069651,
            1.1258706467661694,
            2.2356965174129355,
            0.9043905472636815,
        ]

        np.testing.assert_allclose(dcfs, expected, rtol=1e-5)

    def test_min_dcf(self):
        dcfs = []

        dcfs.append(
            dcf(
                llr=self.commedia_llr_infpar,
                y_val=self.commedia_labels_infpar,
                pi=0.5,
                Cf_n=1,
                Cf_p=1,
                strategy="min",
            )
        )

        dcfs.append(
            dcf(
                llr=self.commedia_llr_infpar,
                y_val=self.commedia_labels_infpar,
                pi=0.8,
                Cf_n=1,
                Cf_p=1,
                strategy="min",
            )
        )

        dcfs.append(
            dcf(
                llr=self.commedia_llr_infpar,
                y_val=self.commedia_labels_infpar,
                pi=0.5,
                Cf_n=10,
                Cf_p=1,
                strategy="min",
            )
        )

        dcfs.append(
            dcf(
                llr=self.commedia_llr_infpar,
                y_val=self.commedia_labels_infpar,
                pi=0.8,
                Cf_n=1,
                Cf_p=10,
                strategy="min",
            )
        )

        expected = [
            0.5061442786069652,
            0.7515422885572139,
            0.8415422885572139,
            0.70931592039801,
        ]

        np.testing.assert_allclose(dcfs, expected, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
