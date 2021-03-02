# hw4-tester

Tests for CS540 Spring 2021 HW4: Clustering

## Changes

### V1.4
 - make float support optional (still needs to remove NaN/inf)

### V1.3
 - clarify error messages

### V1.2
 - test more than 20 points

### V1.1.1
 - check for latest version

### V1.1
 - allow for either `np.matrix` or `np.array` return value from `hac`
 - add test to see if `hac` filters out invalid points

## Usage

Download [test.py](test.py), [Random_Test.csv](Random_Test.csv), and [Tiebreak_Test.csv](Tiebreak_Test.csv) and move them into the directory that contains `pokemon_stats.py` and `Pokemon.csv`

The contents of your directory should look like this:

```shell
$ tree
.
├── pokemon_stats.py
├── Pokemon.csv
├── test.py
├── Random_Test.csv
└── Tiebreak_Test.csv
```

To run the tests, do

```python
$ python3 test.py
```

Ideally, you should be running `test.py` using your terminal as this README describes. If you have an issue, first try running it that way. However, provided that `test.py`, `pokemon_stats.py`, and the 3 csvs are all in the same directory, it should work if you do `%run test.py` in Jupyter, or run it the same way you would run `pokemon_stats.py` in your editor (VS Code, Pycharm, Sublime, etc).

### These tests _do not_ check for `imshow_hac`. They only test `load_data`, `calculate_x_y`, `hac`, and `random_x_y`

## Disclaimer

These tests are not endorsed or created by anyone working in an official capacity with UW Madison or any staff for CS540. The tests are make by students, for students.

By running `test.py`, you are executing code you downloaded from the internet. Back up your files and take a look at what you are running first.

If you have comments or questions, create an issue at [https://github.com/CS540-testers-SP21/hw3-tester/issues](https://github.com/CS540-testers-SP21/hw4-tester/issues) or ask in our discord at [https://discord.gg/RDFNsAxgCQ](https://discord.gg/RDFNsAxgCQ).
