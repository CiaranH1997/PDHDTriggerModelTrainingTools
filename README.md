# trainingtoolsPythonPackage

A custom python package that holds all the python functions used for loading and minipulating data for an ML-based ProtoDUNE trigger algorithm.

## Installation

To install the package editable, run

```
pip install -e PATH/TO/PACKAGE
```

where path/to/package is the folder containing the `pyproject.toml` file

To install on CERN Swan, run

```
Git clone https://github.com/CiaranH1997/PDHDTriggerModelTrainingTools.git
```

or `git pull` to update the package if it already exists on lxplus.

```
pip install --user PDHDTriggerModelTrainingTools/
```

to install the package in the Swan environment. When starting the Swan session, ensure the option is selected to allow the use of external packages.
