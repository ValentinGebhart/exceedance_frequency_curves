"""
Test util functions.
"""

import unittest

import numpy as np


import exceedance_curve_tools.exceedance_curves as cec

# from exceedance_curve_tools import utils


def dummy_exceedance_curve():
    """Create a dummy exceedance curve."""

    values = np.array([1, 10, 100])
    exceedance_frequencies = np.array([1.0, 0.1, 0.01])
    return cec.ExceedanceCurve(
        values, exceedance_frequencies, time_unit="year", value_unit="CHF"
    )


class TestExceedanceFrequencyClass(unittest.TestCase):
    """Test exceedance frequency class."""

    def test_aai_functions(self):
        """Test AAI calculation."""
        ex_curve = dummy_exceedance_curve()
        frequencies = np.array([0.9, 0.09, 0.01])
        np.testing.assert_almost_equal(
            ex_curve.average_annual_impact(), np.sum(frequencies * ex_curve.values)
        )

    def test_aai_one_impact_per_year(self):
        """Test AAI calculation where only one impact can occur every year."""
        ex_curve = dummy_exceedance_curve()
        ex_probs = 1 - np.exp(-ex_curve.exceedance_frequencies)
        ex_probs = np.append(ex_probs, 0)
        probs = np.diff(ex_probs[::-1])[::-1]

        np.testing.assert_almost_equal(
            ex_curve.average_annual_impact(coincidence_fraction=1),
            np.sum(probs * ex_curve.values),
        )


class TestSampling(unittest.TestCase):
    """Test sampling functions."""

    def test__exceedance_probabilities_agg_from_sample(self):
        """Test estimation of exceedance probabilities from samples."""

        sampled_values = np.array(
            [
                [1, 3],
                [2, 5],
                [2, 2],
                [3, 4],
                [0, 0],
                [0, 3],
            ]
        ).T

        nonzero_agg_values_sum, ex_probs_sum = (
            cec._exceedance_probabilities_agg_from_sample(  # pylint: disable=protected-access
                sampled_values, sum
            )
        )
        nonzero_agg_values_max, ex_probs_max = (
            cec._exceedance_probabilities_agg_from_sample(  # pylint: disable=protected-access
                sampled_values, max
            )
        )
        np.testing.assert_almost_equal(nonzero_agg_values_sum, [3, 4, 7])
        np.testing.assert_almost_equal(ex_probs_sum, np.array([5, 4, 2]) / 6)
        np.testing.assert_almost_equal(nonzero_agg_values_max, [2, 3, 4, 5])
        np.testing.assert_almost_equal(ex_probs_max, np.array([5, 4, 2, 1]) / 6)


class TestCombiningFunctions(unittest.TestCase):
    """test combining functions."""

    def test_combine_exceedance_curves_sum_exact(self):
        """Test combining exceedance curves using sum (exact)."""
        ex_curve1 = dummy_exceedance_curve()
        ex_curve2 = dummy_exceedance_curve()
        for cf in [1, 1 / 12]:
            combined_sum = cec.combine_exceedance_curves(
                [ex_curve1, ex_curve2],
                coincidence_fraction=cf,
                use_sampling=False,
                value_resolution=1,
            )

            prob_100 = 1 - np.exp(-ex_curve1.exceedance_frequencies[-1] * cf)
            prob_10 = 1 - np.exp(-ex_curve1.exceedance_frequencies[-2] * cf) - prob_100
            prob_200 = prob_100**2
            prob_110 = prob_100 * prob_10 * 2
            exfreq_200 = -np.log(1 - prob_200) / cf
            exfreq_110 = -np.log(1 - prob_200 - prob_110) / cf

            # test highest and second highest agg events
            np.testing.assert_almost_equal(combined_sum.values[-1], 200)
            np.testing.assert_almost_equal(combined_sum.values[-91], 110)
            np.testing.assert_almost_equal(
                combined_sum.exceedance_frequencies[-1], exfreq_200
            )
            np.testing.assert_almost_equal(
                combined_sum.exceedance_frequencies[-91], exfreq_110
            )
            np.testing.assert_almost_equal(
                combined_sum.exceedance_frequencies[-20], exfreq_200
            )
            np.testing.assert_almost_equal(
                combined_sum.exceedance_frequencies[-95], exfreq_110
            )

    def test_combine_exceedance_curves_max_exact(self):
        """Test combining exceedance curves using max (exact)."""
        ex_curve1 = dummy_exceedance_curve()
        ex_curve2 = dummy_exceedance_curve()
        combined_curve_max = cec.combine_exceedance_curves(
            [ex_curve1, ex_curve2],
            coincidence_fraction=1,
            use_sampling=False,
            value_resolution=1,
        )

        prob_less_than_100 = np.exp(-ex_curve1.exceedance_frequencies[-1])
        prob_agg_less_than_100 = prob_less_than_100**2
        exfreq_100 = -np.log(prob_agg_less_than_100)
        # test max aggreagation
        combined_curve_max = cec.combine_exceedance_curves(
            [ex_curve1, ex_curve2],
            coincidence_fraction=1,
            use_sampling=False,
            value_resolution=1,
            aggregation_method=max,
        )

        np.testing.assert_almost_equal(combined_curve_max.values[-1], 100)
        np.testing.assert_almost_equal(
            combined_curve_max.exceedance_frequencies[-1], exfreq_100
        )

    def test_combine_exceedance_curves_sum_sampling(self):
        """Test combining exceedance curves using sum (sampling)."""
        ex_curve1 = dummy_exceedance_curve()
        ex_curve2 = dummy_exceedance_curve()
        for cf in [1, 1 / 12]:
            combined_sum_exact = cec.combine_exceedance_curves(
                [ex_curve1, ex_curve2],
                coincidence_fraction=cf,
                use_sampling=False,
                value_resolution=1,
            )
            combined_sum_sampling = cec.combine_exceedance_curves(
                [ex_curve1, ex_curve2], coincidence_fraction=cf, n_samples=5000
            )
            index_110_exact = np.where(combined_sum_exact.values == 11)[0][0]
            index_110_sampling = np.where(combined_sum_sampling.values == 11)[0][0]
            np.testing.assert_almost_equal(
                combined_sum_exact.exceedance_frequencies[index_110_exact],
                combined_sum_sampling.exceedance_frequencies[index_110_sampling],
                decimal=2,
            )


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestExceedanceFrequencyClass)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSampling))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCombiningFunctions))
    # TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSampling))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
