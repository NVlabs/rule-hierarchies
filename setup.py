from setuptools import setup

setup(
    name="Rule Hierarchy",
    version="0.0.1",
    description="Create STL rule hierarchies and perform operations with it",
    author="Sushant Veer",
    author_email="sveer@nvidia.com",
    packages=["rule_hierarchy"],
    install_requires=[
        "numpy",
        "stlcg @ git+https://github.com/StanfordASL/stlcg.git@dev",
        "torch",
        "matplotlib",
        "termcolor",
    ],
)
