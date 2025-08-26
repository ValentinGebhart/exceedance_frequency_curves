"""
Test util functions.
"""

import unittest

import numpy as np

from exceedance_curve_tools import utils


class TestFrequencyFunc(unittest.TestCase):
    """Test frequency and probability conversion functions."""

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

    def test_frequency_from_exceedance_frequency(self):
        """test converting exceedance frequency to frequency"""
        exceedance_frequency = np.array(
            [
                [1.0, 0.8, 0.5, 0.1],
                [0.001, 0.0003, 0.0003, 0.0002],
                [300, 10, 1, 0],
            ]
        )
        np.testing.assert_array_almost_equal(
            utils.frequency_from_exceedance_frequency(exceedance_frequency),
            np.array(
                [
                    [0.2, 0.3, 0.4, 0.1],
                    [0.0007, 0.0, 0.0001, 0.0002],
                    [290, 9, 1, 0],
                ]
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

    def test_frecquency_from_exceedance_frequency(self):
        """Test frequency from exceedance frequency."""
        exceedance_frequency = np.array(
            [
                [1.0, 0.8, 0.5, 0.1],
                [0.001, 0.0003, 0.0003, 0.0002],
                [300, 10, 1, 0],
            ]
        )
        np.testing.assert_array_almost_equal(
            utils.frequency_from_exceedance_frequency(exceedance_frequency),
            np.array(
                [
                    [0.2, 0.3, 0.4, 0.1],
                    [0.0007, 0.0, 0.0001, 0.0002],
                    [290, 9, 1, 0],
                ]
            ),
        )


class TestRounding(unittest.TestCase):
    """Test rounding functions."""

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


class TestSampling(unittest.TestCase):
    """Test sampling functions."""

    def test_get_correlated_quantiles(self):
        """Test getting correlated quantiles."""
        dimension = 2
        n_samples = 10000
        correlation = 0.5
        samples = utils.get_correlated_quantiles(dimension, correlation, n_samples)

        # Test shape of output
        np.testing.assert_equal(samples.shape, (n_samples, dimension))

        np.testing.assert_almost_equal(
            np.corrcoef(samples[:, 0], samples[:, 1])[0, 1],
            correlation,
            decimal=1,
        )

        samples = utils.get_correlated_quantiles(dimension, -correlation, n_samples)
        np.testing.assert_almost_equal(
            np.corrcoef(samples[:, 0], samples[:, 1])[0, 1],
            -correlation,
            decimal=1,
        )

        # test invlaid correlation input
        with self.assertRaises(ValueError):
            utils.get_correlated_quantiles(dimension, 1.1, 10)

        with self.assertRaises(ValueError):
            utils.get_correlated_quantiles(dimension, -1.1, 10)


class TestMisc(unittest.TestCase):
    """Test remaining functions."""

    def test_fill_edges(self):
        """Test rounding to array."""
        # generate test array
        arr = np.array(
            [
                [1.0, 2.0, np.NaN, 5.0, 3.0],
                [np.NaN, np.NaN, 4.0, np.NaN, np.NaN],
                [np.NaN, 1.0, np.NaN, np.NaN, 3.0],
                [1.0, 2.0, np.NaN, np.NaN, np.NaN],
                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
            ]
        )

        filled_edges = np.array(
            [
                [1.0, 2.0, np.NaN, 5.0, 3.0],
                [4.0, 4.0, 4.0, 4.0, 4.0],
                [1.0, 1.0, np.NaN, np.NaN, 3.0],
                [1.0, 2.0, 2.0, 2.0, 2.0],
                [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
            ]
        )
        for i in range(len(arr)):
            np.testing.assert_array_equal(utils.fill_edges(arr[i]), filled_edges[i])


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFrequencyFunc)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMisc))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRounding))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSampling))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
