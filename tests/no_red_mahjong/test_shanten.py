import unittest

import jax
import jax.numpy as jnp
import json
import os

from mahjax.no_red_mahjong.shanten import Shanten

jitted_number = jax.jit(Shanten.number)


class TestShanten(unittest.TestCase):
    def test_shanten(self):
        # Load test data
        test_file = os.path.join(os.path.dirname(__file__), "assets/shanten.json")
        with open(test_file, "r") as f:
            data = json.load(f)

        for shanten_name, content in data.items():
            with self.subTest(shanten=shanten_name):
                hand = jnp.int32(content["hand"])
                num_tiles = jnp.int32(content["num_tiles"])
                expected_shanten = jnp.int32(content["shanten"])
                if num_tiles == 14:
                    shanten = jitted_number(hand)
                else:
                    shanten = jitted_number(hand)
                self.assertEqual(shanten, expected_shanten,
                                 f"Shanten mismatch for {shanten_name}: expected {expected_shanten}, got {shanten}")
                print(shanten_name, "passed")
