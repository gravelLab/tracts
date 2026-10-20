from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()


setup(
    # Automatically find all packages and subpackages, except the deprecated legacy module (tracts 1.0),
    # which is kept in the repository for reference but is not installed.
    packages=find_packages(exclude=["tracts.legacy", "tracts.legacy.*"]),
    install_requires=parse_requirements('requirements.txt'),
)
