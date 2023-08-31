from distutils.core import setup

setup(
    name='Tracts',
    version='2.0.2-beta',
    author="Simon Gravel, Victor Yee",
    author_email="simon.gravel@mcgill.ca, aaron.krim-yee@mcgill.ca",
    packages=['tracts',],
    license='MIT',
    long_description='A set of classes and definitions used to model migration histories based on ancestry tracts in admixed individuals. Time-dependent gene-flow from multiple populations can be modeled.',
    python_requires=">=3.6",
    install_requires=[
        "numpy >=1.12.1",
        "matplotlib",
        "scipy",
        "ruamel.yaml"
        
    ],
)