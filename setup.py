from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()


setup(
    name='Tracts',
    version='2.0.3',
    author="Javier González-Delgado, Simon Gravel",
    author_email="javier.gonzalez-delgado@ensai.fr, simon.gravel@mcgill.ca",
    packages=find_packages(),  # Automatically find all packages and subpackages
    license='MIT',
    long_description='A set of tools to model migration histories based on'
                     ' ancestry tracts in admixed individuals. Time-dependent and sex-biased gene-flow from multiple populations'
                     ' can be modeled.',
    python_requires=">=3.9",
    install_requires=parse_requirements('requirements.txt'),
)
