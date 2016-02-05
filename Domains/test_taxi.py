from unittest import TestCase
from Taxi import Taxi
import numpy as np


class TestTaxi(TestCase):
    def test_is_terminal(self):
        a = Taxi()
        self.assertTrue(a.is_terminal(np.array([0, 0, 0, 0])))
        self.assertTrue(a.is_terminal(np.array([1, 1, 0, 0])))
        self.assertTrue(a.is_terminal(np.array([2, 2, 0, 0])))
        self.assertTrue(a.is_terminal(np.array([3, 3, 0, 0])))

    def test_can_move(self):
        a = Taxi()
        self.assertFalse(a.can_move([0, 1], [0, 2]))
        self.assertFalse(a.can_move([0, 2], [0, 1]))
        self.assertFalse(a.can_move([1, 1], [1, 2]))
        self.assertFalse(a.can_move([1, 2], [1, 1]))

        self.assertFalse(a.can_move([3, 0], [3, 1]))
        self.assertFalse(a.can_move([3, 1], [3, 0]))
        self.assertFalse(a.can_move([4, 0], [4, 1]))
        self.assertFalse(a.can_move([4, 1], [4, 0]))

        self.assertFalse(a.can_move([3, 2], [3, 3]))
        self.assertFalse(a.can_move([3, 3], [3, 2]))
        self.assertFalse(a.can_move([4, 2], [4, 3]))
        self.assertFalse(a.can_move([4, 3], [4, 2]))

    def test_step(self):
        a = Taxi()
        a._drift_prob = 0.0
        (r, ns, terminal) = a.step(np.array([1, 0, 0, 0]), 0)
        self.assertEqual((r, ns.tolist(), terminal), (-1, [1, 0, 0, 0], False))

        (r, ns, terminal) = a.step(np.array([1, 0, 0, 0]), 1)
        self.assertEqual((r, ns.tolist(), terminal), (-1, [1, 0, 1, 0], False))

        (r, ns, terminal) = a.step(np.array([1, 0, 0, 0]), 2)
        self.assertEqual((r, ns.tolist(), terminal), (-1, [1, 0, 0, 0], False))

        (r, ns, terminal) = a.step(np.array([1, 0, 0, 0]), 3)
        self.assertEqual((r, ns.tolist(), terminal), (-1, [1, 0, 0, 1], False))

        (r, ns, terminal) = a.step(np.array([2, 0, 0, 0]), 4)
        self.assertEqual((r, ns.tolist(), terminal), (-1, [2, 4, 0, 0], False))

        (r, ns, terminal) = a.step(np.array([0, 4, 1, 1]), 4)
        self.assertEqual((r, ns.tolist(), terminal), (-10, [0, 4, 1, 1], False))

        (r, ns, terminal) = a.step(np.array([0, 4, 1, 0]), 5)
        self.assertEqual((r, ns.tolist(), terminal), (-10, [0, 4, 1, 0], False))

        (r, ns, terminal) = a.step(np.array([0, 4, 0, 0]), 5)
        self.assertEqual((r, ns.tolist(), terminal), (20, [0, 0, 0, 0], True))


    def run_all_test(self):
        self.test_can_move()
        self.test_is_terminal()
        self.test_step()





