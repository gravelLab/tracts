Command line instructions
Only run fast test cases: python run_tests.py
Run all test cases: python run_tests.py --run_all_tests
Add more tests: If any test case is slow, append the test name by '_slow' For example, if the test_case3 is slow, the name should be changed to test_case3_slow so that the script can ignore it in fast mode.
class TestExample(unittest.TestCase): def test_case1(self): pass def test_case2(self): pass def test_case3_slow(self): pass
