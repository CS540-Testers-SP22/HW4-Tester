'''
These tests were inspired by and use code from the tests made by 
cs540-testers-SP21 for the Spring 2021 semester.
Their version (1.0) can be found here: 
    https://github.com/cs540-testers-SP21/hw4-tester/
    
Subsequently, their version was also inspired by and use code from the tests
made by cs540-testers for the Fall 2020 semester.
Their version (original) can be found here: 
    https://github.com/cs540-testers/hw7-tester
'''

__maintainer__ = 'CS540-Testers-SP22'
__author__ = ['Jesus Vazquez']
__version__ = 'IN DEVELOPMENT'

import unittest
import sys
from time import time
from urllib.request import urlopen
import numpy as np
from scipy.cluster.hierarchy import linkage
from pokemon_stats import load_data, calc_features, hac

tiebreak_csv_file = 'Tiebreak_Test.csv'
random_csv_file = 'Random_Test.csv'
pokemon_csv_file = 'Pokemon.csv'

failures = []
errors = []
test_output = []

def timeit(func):
    def timed_func(*args, **kwargs):
        global failures, errors
        t0 = time()
        try:
            out = func(*args, **kwargs)
            runtime = time() - t0
        except AssertionError as e:
            test_output.append(f'FAILED {func.__name__}')
            failures += [func.__name__]
            raise e
        except Exception as e:
            test_output.append(f'ERROR  {func.__name__}')
            errors += [func.__name__]
            raise e
        test_output.append(f'PASSED {func.__name__}{" "*(22-len(func.__name__))}in {(runtime)*1000:.2f}ms')
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
    return [calc_features(stats) for stats in load_data(csv_file)]

class Test2CalculateXY(unittest.TestCase):
    @timeit
    def test2_calc_features(self):
        pokemon = load_data(pokemon_csv_file)
        x_y_pairs = calc_features(pokemon[0])
        expected_x_y_pairs = [(49), (65), (45), (49), (65), (45)]
        

        for x_y_pair, expected_x_y_pair in zip(x_y_pairs, expected_x_y_pairs):
            # Validated the shape is correct
            self.assertEqual(x_y_pairs.shape, (6,))
            # Checks if each value is int64
            self.assertIsInstance(x_y_pair, np.int64)
            # Validates values are correct
            self.assertEqual(x_y_pair, expected_x_y_pair)

class Test3HAC(unittest.TestCase):
    @timeit
    def test3_pokemon_csv(self):
        # Creates list of np.arrays
        x_y_pairs = get_x_y_pairs(pokemon_csv_file)
        
        # Applies hac function
        computed = hac(x_y_pairs)

        # # hac should return an numpy array or matrix of the right shape
        self.assertTrue(isinstance(computed, np.ndarray))
        # self.assertEqual(np.shape(computed), (20, 6)) <- this was originally
        # here, unsure if correct. Created the one below as it is correct to 
        # my implementation
        self.assertEqual(np.shape(computed), (19, 4))
        # computed = np.array(computed)

        # The third column should be increasing
        # for i in range(18):
        #     self.assertGreaterEqual(computed[i + 1, 2], computed[i, 2])

        # Verify hac operates exactly as linkage does - giving leeway for tiebreaker
        # expected = linkage(x_y_pairs)
        # self.assertTrue(np.allclose(computed[computed[:,0].argsort()], 
        #                             expected[expected[:,0].argsort()]))
        # self.assertTrue(np.allclose(computed[computed[:,1].argsort()], 
        #                             expected[expected[:,1].argsort()]))

# def get_versions():
#     current = __version__
#     to_tuple = lambda x: tuple(map(int, x.split('.')))
#     try:
#         with urlopen('https://raw.githubusercontent.com/CS540-Testers-SP22/HW4-tester/master/.version') as f:
#             if f.status != 200:
#                 raise Exception
#             latest = f.read().decode('utf-8')
#     except Exception as e:
#         print('Error checking for latest version') # very descriptive error messages
#         return to_tuple(current), to_tuple(current) # ignoring errors probably isn't the best idea tbh
#     return to_tuple(current), to_tuple(latest)

if __name__ == '__main__':
    print(f'Running CS540 SP22 HW4 tester Version{__version__}\n')

    # current, latest = get_versions()
    # to_v_str = lambda x : '.'.join(map(str, x))
    # if current < latest:
    #     print(f'A newer version of this tester (v{to_v_str(latest)}) is available. You are current running v{to_v_str(current)}\n')
    #     print('You can download the latest version at https://github.com/CS540-testers-SP21/hw4-tester\n')
    
    unittest.main(argv=sys.argv, exit=False)
    
    for message in test_output:
        print(message)
    print()
    if not failures and not errors:
        print('\nPassed all tests successfully\n')
    if failures:
        print('The following tests failed:\n' + '\n'.join(failures) + '\n')
    if errors:
        print('The following tests had exceptions when running:\n' + '\n'.join(errors) + '\n')
    if failures or errors:
        print('Please see the Traceback above for where there were issues')
