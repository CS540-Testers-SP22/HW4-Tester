'''
These tests are inspired by and use code from the tests made by cs540-testers
for the Fall 2020 semester

Their version can be found here: https://github.com/cs540-testers/hw7-tester/
'''

__maintainer__ = 'CS540-testers-SP21'
__author__ = ['Nicholas Beninato']
__credits__ = ['Harrison Clark', 'Stephen Jasina', 'Saurabh Kulkarni', 'Alex Moon']
__version__ = '1.1'

import unittest
import sys
import math
from time import time
import numpy as np
from scipy.cluster.hierarchy import linkage
from pokemon_stats import load_data, calculate_x_y, hac, random_x_y

tiebreak_csv_file = 'Tiebreak_Test.csv'
random_csv_file = 'Random_Test.csv'
pokemon_csv_file = 'Pokemon.csv'

def timeit(func):
    def timed_func(*args, **kwargs):
        t0 = time()
        out = func(*args, **kwargs)
        print(f'Ran {func.__name__}{" "*(30-len(func.__name__))}in {(time() - t0)*1000:.2f}ms')
        return out
    return timed_func

class Test1LoadData(unittest.TestCase):
    @timeit
    def test1_load_data(self):
        pokemon = load_data(random_csv_file)

        # We should have a list
        self.assertIsInstance(pokemon, list)

        # The elements of the list should be dictionaries
        for element in pokemon:
            self.assertIsInstance(element, dict)

        # We should load exactly 20 pokemon
        self.assertEqual(len(pokemon), 20)

        for row in pokemon:
            self.assertTrue(all(k not in ['Legendary', 'Generation'] for k in row))

        # Check row 13 to make sure it contains what we expect
        row = pokemon[13]
        expected_row = {
            '#': 14,
            'Name': 'name_14',
            'Type 1': 'type_a_14',
            'Type 2': '',
            'Total': 687,
            'HP': 191,
            'Attack': 2,
            'Defense': 181,
            'Sp. Atk': 12,
            'Sp. Def': 108,
            'Speed': 193
        }

        # Check that expected_row is contained in row
        for k, v in expected_row.items():
            self.assertIn(k, row)
            self.assertIsInstance(row[k], type(v))
            self.assertEqual(row[k], v)

        # Check that row contains no extra keys
        for k in row:
            self.assertIn(k, expected_row)

def get_x_y_pairs(csv_file):
    '''
    Take in a csv file name and return a list of (x, y) pairs corresponding to
    the csv file's pokemon
    '''
    return [calculate_x_y(stats) for stats in load_data(csv_file)]

class Test2CalculateXY(unittest.TestCase):
    @timeit
    def test2_calculate_x_y(self):
        x_y_pairs = get_x_y_pairs(random_csv_file)
        expected_x_y_pairs = [(318, 172), (197, 165), (256, 276), (243, 300),
                (272, 256), (125, 403), (280, 362), (374, 85), (326, 554),
                (296, 115), (334, 380), (336, 436), (270, 425), (207, 480),
                (347, 401), (186, 305), (267, 304), (396, 184), (469, 518),
                (414, 223)] # I'm sorry this is ugly

        for x_y_pair, expected_x_y_pair in zip(x_y_pairs, expected_x_y_pairs):
            self.assertIsInstance(x_y_pair, tuple)
            self.assertEqual(x_y_pair, expected_x_y_pair)

class Test3HAC(unittest.TestCase):
    @timeit
    def test3_pokemon_csv(self):
        x_y_pairs = get_x_y_pairs(pokemon_csv_file)
        computed = hac(x_y_pairs)

        # hac should return an numpy array or matrix of the right shape
        self.assertTrue(isinstance(computed, np.ndarray) or isinstance(computed, np.matrix))
        self.assertEqual(np.shape(computed), (19, 4))
        computed = np.array(computed)

        # The third column should be increasing
        for i in range(18):
            self.assertGreaterEqual(computed[i + 1, 2], computed[i, 2])

        # Verify hac operates exactly as linkage does - giving leeway for tiebreaker
        expected = linkage(x_y_pairs)
        self.assertTrue(np.allclose(computed[computed[:,0].argsort()], 
                                    expected[expected[:,0].argsort()]))
        self.assertTrue(np.allclose(computed[computed[:,1].argsort()], 
                                    expected[expected[:,1].argsort()]))

    @timeit
    def test4_randomized(self):
        x_y_pairs = get_x_y_pairs(random_csv_file)
        computed = hac(x_y_pairs)

        # hac should return an numpy array or matrix of the right shape
        self.assertTrue(isinstance(computed, np.ndarray) or isinstance(computed, np.matrix))
        self.assertEqual(np.shape(computed), (19, 4))
        computed = np.array(computed)

        # The third column should be increasing
        for i in range(18):
            self.assertGreaterEqual(computed[i + 1, 2], computed[i, 2])

        # Verify hac operates exactly as linkage does
        expected = linkage(x_y_pairs)
        self.assertTrue(np.all(np.isclose(computed, expected)))

    @timeit
    def test5_filter_finite(self):
        x_y_pairs = [(0,0), (1,1), (math.inf, 9), (9, math.inf),
                     (-math.inf, 2), (2, -math.inf), (3.2, -20),
                     (float("nan"), 0), (0, float("nan")), (7, 7),
                     (float("inf"), 100), (100, float("inf")), (4, 0),
                     (float("-inf"), 1), (1, float("inf")), (4, 0),
                     (0.0, 0.1), (-3.9, 27)]
        computed = hac(x_y_pairs)

        # hac should return an numpy array or matrix of the right shape
        self.assertTrue(isinstance(computed, np.ndarray) or isinstance(computed, np.matrix))
        self.assertEqual(np.shape(computed), (7, 4))
        computed = np.array(computed)

        # The third column should be increasing
        for i in range(6):
            self.assertGreaterEqual(computed[i + 1, 2], computed[i, 2])

        # Verify hac operates exactly as linkage does
        expected = linkage([(0, 0), (1, 1), (3.2, -20), (7, 7),
                            (4, 0), (4, 0), (0.0, 0.1), (-3.9, 27)])

        self.assertTrue(np.all(np.isclose(computed, expected)))

    @timeit
    def test5_tiebreak(self):
        x_y_pairs = get_x_y_pairs(tiebreak_csv_file)
        computed = hac(x_y_pairs)
        expected_cluster_sizes \
                = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 8, 8, 12, 20]

        # chose lowest cluster index for the first position
        # if still tied, chose lowest cluster index for the second position
        for i in range(np.shape(computed)[0]):
            row = np.array(computed[i,:]).flatten()
            self.assertEqual(row[0], 2 * i)
            self.assertEqual(row[1], 2 * i + 1)
            self.assertEqual(row[2], 0)
            self.assertEqual(row[3], expected_cluster_sizes[i])

class Test4RandomXY(unittest.TestCase):
    @timeit
    def test6_random_x_y(self):
        # empty list
        self.assertEqual(random_x_y(0), [])
        
        # various values of m
        for m in range(0, 11, 2):
            x_y_pairs = random_x_y(2**m)
            # x_y_pairs is a list with length 2**m
            self.assertIsInstance(x_y_pairs, list)
            self.assertEqual(len(x_y_pairs), 2**m)
            # that list contains tuples
            self.assertTrue(all(isinstance(x, tuple) for x in x_y_pairs))
            # those tuples contain ints
            self.assertTrue(all(isinstance(x, int) and isinstance(y, int) for x, y in x_y_pairs))
            # all ints are > 0 and < 360
            self.assertTrue(all(0 < x < 360 and 0 < y < 360 for x, y in x_y_pairs))

if __name__ == '__main__':
    print(f'Running CS540 SP21 HW4 tester v{__version__}')

    unittest.main(argv=sys.argv)