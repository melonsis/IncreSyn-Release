# DADP: Dynamic Attribute Aware Framework for Differentially Private Data Synthesis with Efficient Update

## Introduction

This code-base contains two example of DADP-Combined mechanisms, which combined DADP framework to state-of-art graphic model based data synthesis mechanisms, the MWEM+PGM and AIM.

For more details of Private-PGM, please visit [Private-PGM](https://github.com/ryan112358/private-pgm).

These files also have two additional dependencies: [Ektelo](https://github.com/ektelo/ektelo) and [autograd](https://github.com/HIPS/autograd).

## File structure

* mechanisms - Contains DADP-Combined mechanisms which modified for initialization phase of DADP.
* da-mechanisms - Contains DADP-combined mechanisms for update phase, and some dataset utility tools.
* data - Contains datasets, selected cliques which produced in initialization phase and preferred cliques.
* src = Contains some dependencies of PGM-Based mechanisms.
* UDF - Contains example DADP-combined UDFs. 

## Usage

1. Solve the dependencies with ```requirements.txt```. Note that we only supporting Python 3.
```
$ pip install -r requirements.txt
```
2. Export the ```src``` file to path. For example, in windows, you may using:
```
$Env:PYTHONPATH += ";X:\DADP\src"
```
1. Run the mechanism under ```mechanisms``` for initialization phase, then run the corresponding mechanism under ```da-mechanisms```.

## Utilities usage
* ```\da-mechanisms\werror.py``` - Calculate the workload error for a given preferred attributes file ```prefer.csv```.
