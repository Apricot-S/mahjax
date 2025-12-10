# tests/no_red_mahjong/test_meld.py
import unittest
from mahjax.no_red_mahjong.meld import Meld
from mahjax.no_red_mahjong.action import Action
import jax.numpy as jnp
import jax

class TestMeld(unittest.TestCase):
    def setUp(self):
        self.pon_1m = Meld.init(Action.PON, 0, 1)  # 111m
        self.pon_2s = Meld.init(Action.PON, 19, 2)  # 222s
        self.pon_white = Meld.init(Action.PON, 31, 2)  # WWW
        self.chi_l_123p = Meld.init(Action.CHI_L, 9, 3)  # [1]23p
        self.chi_m_123p = Meld.init(Action.CHI_M, 10, 3)  # 1[2]3p
        self.chi_r_123p = Meld.init(Action.CHI_R, 11, 3)  # 12[3]p
        self.chi_l_234p = Meld.init(Action.CHI_L, 10, 3)  # [2]34p
        self.chi_m_234p = Meld.init(Action.CHI_M, 11, 3)  # 2[3]4p
        self.chi_r_234p = Meld.init(Action.CHI_R, 12, 3)  # 23[4]p
        self.chi_l_789p = Meld.init(Action.CHI_L, 15, 3)  # [7]89p
        self.chi_m_789p = Meld.init(Action.CHI_M, 16, 3)  # 7[8]9p
        self.chi_r_789p = Meld.init(Action.CHI_R, 17, 3)  # 78[9]p
        self.closed_kan_1m = Meld.init(34, 0, 0)  # 1111m
        self.closed_kan_2m = Meld.init(35, 1, 0)  # 2222m
        self.open_kan_1m = Meld.init(Action.OPEN_KAN, 0, 1)  # 1111m
        self.open_kan_2m = Meld.init(Action.OPEN_KAN, 1, 2)  # 2222m
        self.added_kan_1m = Meld.init(34, 0, 1)  # 1111m
        self.added_kan_2m = Meld.init(35, 1, 2)  # 2222m
        self.empty = Meld.empty()

    def test_target(self):
        self.assertEqual(int(Meld.target(self.pon_2s)), 19)  # 222s -> 19
        self.assertEqual(int(Meld.target(self.chi_l_123p)), 9)  # [1]23p -> 9
        self.assertEqual(int(Meld.target(self.chi_m_123p)), 10)  # 1[2]3p -> 10
        self.assertEqual(int(Meld.target(self.chi_r_123p)), 11)  # 12[3]p -> 11
        self.assertEqual(int(Meld.target(self.chi_l_234p)), 10)  # [2]34p -> 10
        self.assertEqual(int(Meld.target(self.chi_m_234p)), 11)  # 2[3]4p -> 11
        self.assertEqual(int(Meld.target(self.chi_r_234p)), 12)  # 23[4]p -> 12
        self.assertEqual(int(Meld.target(self.chi_l_789p)), 15)  # [7]89p -> 15
        self.assertEqual(int(Meld.target(self.chi_m_789p)), 16)  # 7[8]9p -> 16
        self.assertEqual(int(Meld.target(self.chi_r_789p)), 17)  # 78[9]p -> 17
        self.assertEqual(int(Meld.target(self.closed_kan_1m)), 0)  # 1111m -> 0
        self.assertEqual(int(Meld.target(self.closed_kan_2m)), 1)  # 2222m -> 1

    def test_action(self):
        self.assertEqual(int(Meld.action(self.pon_1m)), Action.PON)
        self.assertEqual(int(Meld.action(self.pon_2s)), Action.PON)
        self.assertEqual(int(Meld.action(self.pon_white)), Action.PON)
        self.assertEqual(int(Meld.action(self.chi_l_123p)), Action.CHI_L)
        self.assertEqual(int(Meld.action(self.chi_m_123p)), Action.CHI_M)
        self.assertEqual(int(Meld.action(self.chi_r_123p)), Action.CHI_R)
        self.assertEqual(int(Meld.action(self.chi_l_234p)), Action.CHI_L)
        self.assertEqual(int(Meld.action(self.chi_m_234p)), Action.CHI_M)
        self.assertEqual(int(Meld.action(self.chi_r_234p)), Action.CHI_R)
        self.assertEqual(int(Meld.action(self.chi_l_789p)), Action.CHI_L)
        self.assertEqual(int(Meld.action(self.chi_m_789p)), Action.CHI_M)
        self.assertEqual(int(Meld.action(self.chi_r_789p)), Action.CHI_R)
        self.assertEqual(int(Meld.action(self.closed_kan_1m)), 34)
        self.assertEqual(int(Meld.action(self.closed_kan_2m)), 35)
        self.assertEqual(int(Meld.action(self.open_kan_1m)), Action.OPEN_KAN)
        self.assertEqual(int(Meld.action(self.open_kan_2m)), Action.OPEN_KAN)

    def test_src(self):
        self.assertEqual(int(Meld.src(self.pon_2s)), 2)  # from mid
        self.assertEqual(int(Meld.src(self.chi_l_123p)), 3)  # from left
        self.assertEqual(int(Meld.src(self.closed_kan_1m)), 0)  # from self
        self.assertEqual(int(Meld.src(self.open_kan_2m)), 2)  # from mid
        self.assertEqual(int(Meld.src(self.added_kan_1m)), 1)  # from right

    def test_to_str(self):
        # Skip; formatting depends on locale/choice
        pass

    def test_suited_pung(self):
        self.assertEqual(int(Meld.suited_pung(self.pon_2s)), 1 << 19)
        self.assertEqual(int(Meld.suited_pung(self.chi_l_123p)), 0)
        self.assertEqual(int(Meld.suited_pung(self.chi_m_234p)), 0)
        self.assertEqual(int(Meld.suited_pung(self.pon_white)), 0)
        self.assertEqual(int(Meld.suited_pung(self.closed_kan_1m)), 1 << 0)
        self.assertEqual(int(Meld.suited_pung(self.open_kan_2m)), 1 << 1)

    def test_chow(self):
        self.assertEqual(int(Meld.chow(self.chi_l_123p)), 1 << (9 - (Action.CHI_L - Action.CHI_L)))
        self.assertEqual(int(Meld.chow(self.chi_m_234p)), 1 << (11 - (Action.CHI_M - Action.CHI_L)))
        self.assertEqual(int(Meld.chow(self.pon_2s)), 0)
        self.assertEqual(int(Meld.chow(self.pon_white)), 0)
        self.assertEqual(int(Meld.chow(self.closed_kan_1m)), 0)
        self.assertEqual(int(Meld.chow(self.open_kan_2m)), 0)


    def test_is_kan(self):
        self.assertEqual(bool(Meld.is_kan(self.chi_l_123p)), False)  # [1]23p -> False
        self.assertEqual(bool(Meld.is_kan(self.chi_m_234p)), False)  # 2[3]4p -> False
        self.assertEqual(bool(Meld.is_kan(self.chi_l_789p)), False)  # [7]89p -> False
        self.assertEqual(bool(Meld.is_kan(self.pon_1m)), False)  # 111m -> False
        self.assertEqual(bool(Meld.is_kan(self.pon_2s)), False)  # 222s -> False
        self.assertEqual(bool(Meld.is_kan(self.pon_white)), False)  # WWW -> False
        self.assertEqual(bool(Meld.is_kan(self.closed_kan_1m)), True)  # 1111m -> True
        self.assertEqual(bool(Meld.is_kan(self.closed_kan_2m)), True)  # 2222m -> True
        self.assertEqual(bool(Meld.is_kan(self.open_kan_1m)), True)  # 1111m -> True
        self.assertEqual(bool(Meld.is_kan(self.open_kan_2m)), True)  # 2222m -> True


    def test_is_closed_kan(self):
        self.assertEqual(bool(Meld.is_closed_kan(self.chi_l_123p)), False)  # [1]23p -> False
        self.assertEqual(bool(Meld.is_closed_kan(self.chi_m_234p)), False)  # 2[3]4p -> False
        self.assertEqual(bool(Meld.is_closed_kan(self.chi_l_789p)), False)  # [7]89p -> False
        self.assertEqual(bool(Meld.is_closed_kan(self.pon_1m)), False)  # 111m -> False
        self.assertEqual(bool(Meld.is_closed_kan(self.pon_2s)), False)  # 222s -> False
        self.assertEqual(bool(Meld.is_closed_kan(self.pon_white)), False)  # WWW -> False
        self.assertEqual(bool(Meld.is_closed_kan(self.closed_kan_1m)), True)  # 1111m -> True
        self.assertEqual(bool(Meld.is_closed_kan(self.closed_kan_2m)), True)  # 2222m -> True
        self.assertEqual(bool(Meld.is_closed_kan(self.open_kan_1m)), False)  # 1111m -> False
        self.assertEqual(bool(Meld.is_closed_kan(self.open_kan_2m)), False)  # 2222m -> False
        self.assertEqual(bool(Meld.is_closed_kan(self.added_kan_1m)), False)  # 1111m -> False
        self.assertEqual(bool(Meld.is_closed_kan(self.added_kan_2m)), False)  # 2222m -> False

    def test_is_added_kan(self):
        self.assertEqual(bool(Meld.is_added_kan(self.chi_l_123p)), False)  # [1]23p -> False
        self.assertEqual(bool(Meld.is_added_kan(self.chi_m_234p)), False)  # 2[3]4p -> False
        self.assertEqual(bool(Meld.is_added_kan(self.chi_l_789p)), False)  # [7]89p -> False
        self.assertEqual(bool(Meld.is_added_kan(self.pon_1m)), False)  # 111m -> False
        self.assertEqual(bool(Meld.is_added_kan(self.pon_2s)), False)  # 222s -> False
        self.assertEqual(bool(Meld.is_added_kan(self.pon_white)), False)  # WWW -> False
        self.assertEqual(bool(Meld.is_added_kan(self.closed_kan_1m)), False)  # 1111m -> False
        self.assertEqual(bool(Meld.is_added_kan(self.closed_kan_2m)), False)  # 2222m -> False
        self.assertEqual(bool(Meld.is_added_kan(self.open_kan_1m)), False)  # 1111m -> False
        self.assertEqual(bool(Meld.is_added_kan(self.open_kan_2m)), False)  # 2222m -> False
        self.assertEqual(bool(Meld.is_added_kan(self.added_kan_1m)), True)  # 1111m -> True
        self.assertEqual(bool(Meld.is_added_kan(self.added_kan_2m)), True)  # 2222m -> True

    def test_is_chi(self):
        self.assertEqual(bool(Meld.is_chi(self.chi_l_123p)), True)  # [1]23p -> True
        self.assertEqual(bool(Meld.is_chi(self.chi_m_234p)), True)  # 2[3]4p -> True
        self.assertEqual(bool(Meld.is_chi(self.chi_l_789p)), True)  # [7]89p -> True
        self.assertEqual(bool(Meld.is_chi(self.pon_1m)), False)  # 111m -> False
        self.assertEqual(bool(Meld.is_chi(self.pon_2s)), False)  # 222s -> False
        self.assertEqual(bool(Meld.is_chi(self.pon_white)), False)  # WWW -> False
        self.assertEqual(bool(Meld.is_chi(self.closed_kan_1m)), False)  # 1111m -> False
        self.assertEqual(bool(Meld.is_chi(self.closed_kan_2m)), False)  # 2222m -> False

    def test_is_pon(self):
        self.assertEqual(bool(Meld.is_pon(self.pon_1m)), True)  # 111m -> True
        self.assertEqual(bool(Meld.is_pon(self.pon_2s)), True)  # 222s -> True
        self.assertEqual(bool(Meld.is_pon(self.pon_white)), True)  # WWW -> True
        self.assertEqual(bool(Meld.is_pon(self.chi_l_123p)), False)  # [1]23p -> False
        self.assertEqual(bool(Meld.is_pon(self.chi_m_234p)), False)  # 2[3]4p -> False
        self.assertEqual(bool(Meld.is_pon(self.chi_l_789p)), False)  # [7]89p -> False
        self.assertEqual(bool(Meld.is_pon(self.closed_kan_1m)), False)  # 1111m -> False
        self.assertEqual(bool(Meld.is_pon(self.closed_kan_2m)), False)  # 2222m -> False

    def test_is_outside(self):
        self.assertEqual(bool(Meld.is_outside(self.chi_l_123p)), False)  # [1]23p -> False
        self.assertEqual(bool(Meld.is_outside(self.chi_m_234p)), False)  # 2[3]4p -> False
        self.assertEqual(bool(Meld.is_outside(self.chi_l_789p)), False)  # [7]89p -> False
        self.assertEqual(bool(Meld.is_outside(self.pon_1m)), True)  # 111m -> True
        self.assertEqual(bool(Meld.is_outside(self.pon_2s)), False)  # 222s -> False
        self.assertEqual(bool(Meld.is_outside(self.pon_white)), True)  # WWW -> True
        self.assertEqual(bool(Meld.is_outside(self.closed_kan_1m)), True)  # 1111m -> True
        self.assertEqual(bool(Meld.is_outside(self.open_kan_1m)), True)  # 1111m -> True
        self.assertEqual(bool(Meld.is_outside(self.open_kan_2m)), False)  # 2222m -> False

    def test_has_outside(self):
        self.assertEqual(bool(Meld.has_outside(self.chi_l_123p)), True)  # [1]23p -> True
        self.assertEqual(bool(Meld.has_outside(self.chi_m_123p)), True)  # 1[2]3p -> True
        self.assertEqual(bool(Meld.has_outside(self.chi_m_234p)), False)  # 2[3]4p -> False
        self.assertEqual(bool(Meld.has_outside(self.chi_l_789p)), True)  # [7]89p -> True
        self.assertEqual(bool(Meld.has_outside(self.pon_1m)), True)  # 111m -> True
        self.assertEqual(bool(Meld.has_outside(self.pon_2s)), False)  # 222s -> False
        self.assertEqual(bool(Meld.has_outside(self.pon_white)), True)  # WWW -> True
        self.assertEqual(bool(Meld.has_outside(self.closed_kan_1m)), True)  # 1111m -> True
        self.assertEqual(bool(Meld.has_outside(self.open_kan_2m)), False)  # 2222m -> False

    def test_fu(self):
        self.assertEqual(int(Meld.fu(self.pon_1m)), 4)  # pon is 4 fu
        self.assertEqual(int(Meld.fu(self.pon_2s)), 2)  # pon is 2 fu
        self.assertEqual(int(Meld.fu(self.pon_white)), 4)  # honors are 4 fu
        self.assertEqual(int(Meld.fu(self.chi_l_123p)), 0)  # chi is 0 fu
        self.assertEqual(int(Meld.fu(self.chi_m_234p)), 0)  # chi is 0 fu
        self.assertEqual(int(Meld.fu(self.closed_kan_1m)), 32)  # closed kan is 32 fu
        self.assertEqual(int(Meld.fu(self.closed_kan_2m)), 16)  # closed kan is 16 fu
        self.assertEqual(int(Meld.fu(self.open_kan_1m)), 16)  # open kan is 16 fu
        self.assertEqual(int(Meld.fu(self.open_kan_2m)), 8)  # open kan is 8 fu

    # --- additional: empty verification ---
    def test_empty(self):
        m = self.empty
        self.assertTrue(bool(Meld.is_empty(m)))
        self.assertEqual(int(Meld.target(m)), -1)  # empty target is -1
        self.assertEqual(int(Meld.src(m)), -1)  # empty src is -1
        self.assertEqual(int(Meld.action(m)), -1)  # empty action is -1
        self.assertFalse(bool(Meld.is_pon(m)))  # empty is not pon
        self.assertFalse(bool(Meld.is_chi(m)))  # empty is not chi
        self.assertFalse(bool(Meld.is_kan(m)))  # empty is not kan
        self.assertEqual(int(Meld.suited_pung(m)), 0)  # empty is not suited pung
        self.assertEqual(int(Meld.chow(m)), 0)  # empty is not chow
        self.assertEqual(int(Meld.is_outside(m)), 0)  # empty is not outside
        self.assertEqual(int(Meld.has_outside(m)), 0)  # empty is not has outside
        self.assertEqual(int(Meld.fu(m)), 0)  # empty is not fu


    def test_prohibitive_tile_type_after_chi(self):
        self.assertEqual(int(Meld.prohibitive_tile_type_after_chi(Action.CHI_L, 0)), 3)  # [1]23m -> 4m
        self.assertEqual(int(Meld.prohibitive_tile_type_after_chi(Action.CHI_R, 8)), 5)  # 67[8]m -> 5m
        self.assertEqual(int(Meld.prohibitive_tile_type_after_chi(Action.CHI_L, 6)), -1)  # [7]89m -> no prohibitive tile
        self.assertEqual(int(Meld.prohibitive_tile_type_after_chi(Action.CHI_R, 2)), -1)  # 12[3]m -> no prohibitive tile

if __name__ == "__main__":
    unittest.main()
