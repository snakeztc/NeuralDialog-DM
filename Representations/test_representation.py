from unittest import TestCase
import numpy as np
from Representation import Representation


class TestRepresentation(TestCase):
    def test_expand_state_space(self):
        state = np.array([[0, 1]])
        limit = np.array([[0, 2], [-1, 3]])
        result = Representation.expand_state_space(state, limit)
        expected_result = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        self.assertTrue(np.array_equal(result, expected_result))
