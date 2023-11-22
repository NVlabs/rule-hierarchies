# Rule Hierarchies

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![NVIDIA Source Code License](https://img.shields.io/badge/license-NSCL-blue.svg)](LICENSE.md)
[![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)](https://www.python.org/downloads/release/python-3916/)

## About
This package allows creation of STL rule hierarchies and computing their rank-preserving rewards, as detailed in this paper: https://arxiv.org/pdf/2212.03323.pdf. 

## Installation
To setup this package:
- clone this repository and then `pip install -e .`

## Details
The core code is in [`rule_hierarchy/rule_hierarchy.py`](rule_hierarchy/rule_hierarchy.py). The `AbstractRuleHierarchy` class is responsible for evaluating the STL rule hierarchy while the `RuleHierarchy` class takes the user-specified hierarchy and transforms it into an STL rule hierarchy. The user-specified rule hierarchy is communicated to `RuleHierarchy` as a list of [`Rule`](rule_hierarchy/rules/rule.py) objects, ordered in decreasing order of importance; some examples of rules are provided in the [`rules`](rule_hierarchy/rules/) directory. 

A demo of how to build a rule hierarchy is provided in [demo/simple_demo.py](demo/simple_demo.py).

## How To Use
The main steps to setting up your custom rule hierarchy are:
- Create a subclass of [`Rule`](rule_hierarchy/rules/rule.py) tailored to the specific rule being implemented in the [`rules`](rule_hierarchy/rules/) directory.
    - The class should have a `as_stl_formula()` and a `prepare_signals()` method which generates an STLCG formula and a properly shaped signal for STLCG to evaluate, respectively.
    - Examples of how to formulate the rule class can be found in the [`rules`](rule_hierarchy/rules/) directory.
    - Include an import to the class in `__init__.py` in [`rules/__init__.py`](rule_hierarchy/rules/__init__.py).
- Create an ordered list of rules, e.g., `rules = [AlwaysGreater(1.0), AlwaysLesser(2.0)]` and pass it to the `RuleHierarchy` class to create a rule hierarchy; see [demo/demo.py](demo/simple_demo.py) for a simple example.

## Demo
A simple demo of how to use the rule hierarchy is presented in [demo/simple_demo.py](demo/simple_demo.py) and a more involved demo that demonstrates planning with a continuous optimizer is provided in [demo/optimization_demo.py](demo/optimization_demo.py).

## Citation
Please cite the relevant paper if you use this code:
```
@article{veer2022receding,
  title={Receding Horizon Planning with Rule Hierarchies for Autonomous Vehicles},
  author={Veer, Sushant and Leung, Karen and Cosner, Ryan and Chen, Yuxiao and Karkus, Peter and Pavone, Marco},
  journal={arXiv preprint arXiv:2212.03323},
  year={2022}
}
```

## Contributors
1. [Sushant Veer](https://sushantveer.github.io/)
2. [Apoorva Sharma](https://research.nvidia.com/person/apoorva-sharma)


## License
The source code is released under the [NSCL licence](LICENSE.md).