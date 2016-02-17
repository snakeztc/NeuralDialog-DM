from unittest import TestCase
from BinarySimulator20q import BinarySimulator20q
import numpy as np


class TestBinarySimulator20q(TestCase):
    def test_init_user(self):
        self.fail()

    def test_s0(self):
        self.fail()

    def test_value_id_2_slot_id(self):
        env = BinarySimulator20q(1)
        env.slot_value_basecntt = np.cumsum([len(val) for val in [[1,2,3], [1], [1,2]]])
        self.assertEqual(env.value_id_2_slot_id(0), 0)
        self.assertEqual(env.value_id_2_slot_id(1), 0)
        self.assertEqual(env.value_id_2_slot_id(2), 0)
        self.assertEqual(env.value_id_2_slot_id(3), 1)
        self.assertEqual(env.value_id_2_slot_id(4), 2)
        self.assertEqual(env.value_id_2_slot_id(5), 2)

    def test_possible_string_actions(self):
        self.fail()

    def test_get_inform(self):
        self.fail()

    def test_search(self):
        self.fail()

    def test_stra2index(self):
        self.fail()

    def test_is_question(self):
        self.fail()

    def test_set_slot_yes(self):
        self.fail()

    def test_set_slot_unknown(self):
        self.fail()

    def test_step(self):
        self.fail()

    def test_is_terminal(self):
        self.fail()
