"""
Test util functions.
"""

import unittest

import numpy as np

import utils


class TestFunc(unittest.TestCase):
    """Test conversion functions."""

    def test_prob_from_exceedance_frequency(self):
        """Test converting exceedance frequencies to probabilities."""

        # generate test execeedance frequencies
        exceedance_frequencies = np.array(
            [
                [0.1, 0.01, 0.001, 0.0001],  # first ex freq list
                [0.2, 0.02, 0.002, 0.0002],  # second ex freq list
            ]
        )

        # test shape of output
        probabilities = utils.prob_from_exceedance_frequency(
            exceedance_frequencies, coincidence_fraction=1
        )
        np.testing.assert_equal(
            (probabilities.shape[0], probabilities.shape[1] - 1),
            exceedance_frequencies.shape,
        )

        # test probability for nothing
        np.testing.assert_array_almost_equal(
            probabilities[:, 0],
            np.exp(-exceedance_frequencies[:, 0]),
        )

        # test normalization
        np.testing.assert_allclose(
            np.sum(probabilities, axis=1),
            np.ones(2),
        )

        # only one array as input
        probabilities_single = utils.prob_from_exceedance_frequency(
            exceedance_frequencies[0], coincidence_fraction=1
        )
        np.testing.assert_array_almost_equal(
            probabilities_single,
            probabilities[0],
        )

        # probabilities with coincidence fraction
        probabilities = utils.prob_from_exceedance_frequency(
            exceedance_frequencies, coincidence_fraction=1 / 12
        )
        np.testing.assert_array_almost_equal(
            probabilities[:, 0],
            np.exp(-exceedance_frequencies[:, 0] * (1 / 12)),
        )

    def test_exceedance_frequency_from_prob(self):
        """Test converting probabilities to exceedance frequencies."""

        # generate test execeedance frequencies
        probabilities = np.array(
            [
                [0.5, 0.3, 0.1, 0.1],  # first probability list
                [0.3, 0.3, 0.2, 0.2],  # second probability list
            ]
        )

        # test shape of output
        exceedance_frequencies = utils.exceedance_frequency_from_prob(
            probabilities, coincidence_fraction=1
        )
        np.testing.assert_equal(
            (exceedance_frequencies.shape[0], exceedance_frequencies.shape[1] + 1),
            probabilities.shape,
        )

        # check inverse of prob_from_exceedance_frequency
        np.testing.assert_array_almost_equal(
            probabilities,
            utils.prob_from_exceedance_frequency(
                exceedance_frequencies, coincidence_fraction=1
            ),
        )

    def test_round_to_array(self):
        """Test rounding to array."""
        # generate test array
        arr = np.array([1.0, 2.0, 4.0, 5.0])

        # test rounding to array
        obj = np.array(
            [
                [1.1, 2.8, 3.9],
                [2.2, 3.3, 4.8],
            ]
        )
        rounded = utils.round_to_array(obj, arr)
        np.testing.assert_array_equal(
            rounded, np.array([[1.0, 2.0, 4.0], [2.0, 4.0, 5.0]])
        )

        # test rounding with a single value
        obj_single = 2.7
        rounded_single = utils.round_to_array(obj_single, arr)
        np.testing.assert_array_equal(rounded_single, 2.0)

        # test error for non-1D array
        with self.assertRaises(ValueError):
            utils.round_to_array(obj, np.array([[1, 2], [3, 4]]))


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFunc)
    # TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAssign))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
