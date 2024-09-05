# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent))
# path to tracts, may have to adjust if the file is moved
import tracts.driver

"""
Tests for component methods of the driver script
"""


def test_chromosomes():
    assert tracts.driver.parse_chromosomes(['1-5', 10, '13-18']) == [1, 2, 3, 4, 5, 10, 13, 14, 15, 16, 17, 18]
