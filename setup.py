from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()


setup(
    packages=find_packages(),  # Automatically find all packages and subpackages
    install_requires=parse_requirements('requirements.txt'),
)
