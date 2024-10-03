from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()


setup(
    name='Tracts',
    version='2.0.3',
    author="Simon Gravel, Victor Yee",
    author_email="simon.gravel@mcgill.ca, aaron.krim-yee@mcgill.ca",
    packages=find_packages(),  # Automatically find all packages and subpackages
    license='MIT',
    long_description='A set of classes and definitions used to model migration histories based on'
                     ' ancestry tracts in admixed individuals. Time-dependent gene-flow from multiple populations'
                     ' can be modeled.',
    python_requires=">=3.9",
    install_requires=parse_requirements('requirements.txt'),
)
