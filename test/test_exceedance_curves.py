"""
Test util functions.
"""

import unittest

# import numpy as np

# import exceedance_curve_tools.exceedance_curves as cec
# from exceedance_curve_tools import utils


class TestFunc(unittest.TestCase):
    """Test conversion functions."""

    def test__combine_two_prob_sets(self):
        """Test combining two probabilistic sets."""
        return

    def test_combine_exceedance_curves(self):
        """Test combining two probabilistic sets."""
        return

    def test__combine_two_prob_sets2(self):
        """Test combining two probabilistic sets."""

        return
        


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFunc)
    # TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAssign))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
