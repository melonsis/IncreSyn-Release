# DADP: Dynamic Attribute Aware Framework for Differentially Private Data Synthesis with Efficient Update

## Introduction

This code-base contains two examples of DADP-Combined mechanisms, which combine the DADP framework with state-of-art graphic model-based data synthesis mechanisms, the MWEM+PGM and AIM.

For more details on Private-PGM, please visit [Private-PGM](https://github.com/ryan112358/private-pgm).

These files also have two additional dependencies: [Ektelo](https://github.com/ektelo/ektelo) and [autograd](https://github.com/HIPS/autograd).

## File structure

* mechanisms - Contains DADP-Combined mechanisms, which are modified for the initialization phase of DADP.
* da-mechanisms - Contains DADP-combined mechanisms for the update phase and some dataset utility tools.
* data - Contains datasets, selected cliques produced in the initialization phase, and preferred cliques.
* src - Contains some dependencies of PGM-Based mechanisms.
* UDF - Contains example DADP-combined UDFs. 

## Usage

1. Solve the dependencies with ```requirements.txt```. Note that we only support Python 3.
```
$ pip install -r requirements.txt
```
2. Export the ```src``` file to path. For example, in Windows, you may use:
```
$Env:PYTHONPATH += ";X:\DADP\src"
```
1. Run the mechanism under ```mechanisms``` for initialization phase, then run the corresponding mechanism under ```da-mechanisms```.

## Utility usage
* ```\da-mechanisms\werror.py``` - Calculate the workload error for a given preferred attributes file ```prefer.csv```.