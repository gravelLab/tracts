from glob import glob
import unittest
import sys
sys.path.append('../')
import tracts


def run_fast_tests(suite):
    """
    Run fast tests in the given test suite.
    :param suite:
    :return:
    """
    for ts in suite:
        for t in ts:
            if t.id().endswith("slow"):
                setattr(t, "setUp", lambda: t.skipTest("slow tests"))
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    fast_mode = True
    if len(sys.argv) == 1:
        fast_mode = True
    elif sys.argv[1] == "--run_all_tests":
        fast_mode = False
    else:
        print(
            "Invalid argument. Run fast tests by default or add '--run_all_tests' flag to run all test cases."
        )
        exit()

    # First we collect all our tests into a single TestSuite object.
    all_tests = unittest.TestSuite()

    testfiles = glob("test_*.py")
    all_test_mods = []
    for file in testfiles:
        module = file.split(".")[0]
        print(module)
        mod = __import__(module)
        all_tests.addTest(mod.suite)
    if fast_mode:
        run_fast_tests(all_tests)
    else:
        unittest.TextTestRunner(verbosity=2).run(all_tests)
