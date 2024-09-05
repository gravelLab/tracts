from setuptools import setup, find_packages

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
    python_requires=">=3.6",
    install_requires=[
        "numpy >=1.12.1",
        "matplotlib",
        "scipy",
        "ruamel.yaml",
        "Sphinx>=7.2.6",
        "sphinx-autodoc-typehints>=1.12.0",
        "sphinxcontrib-napoleon>=0.7",
        "sphinx-book-theme>=1.0.0"
    ],
)
