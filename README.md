# DADP: Dynamic Attribute Aware Framework for Differentially Private Data Synthesis with Efficient Update

## Introduction

This code base contains two construction examples of DADP, which construct with state-of-art iterative graphic model-based data synthesis mechanisms, the MWEM+PGM and AIM.

For more details on Private-PGM, please visit [Private-PGM](https://github.com/ryan112358/private-pgm).

These files also have two additional dependencies: [Ektelo](https://github.com/ektelo/ektelo) and [autograd](https://github.com/HIPS/autograd).

## File structure

* mechanisms - Contains DADP-Combined mechanisms, which are modified for the initialization phase of DADP.
* da-mechanisms - Contains DADP-combined mechanisms for the update phase and some dataset utility tools.
* data - Contains datasets, selected cliques produced in the initialization phase, and preferred cliques.
* src - Contains some dependencies of PGM-Based mechanisms.
* UDF - Contains example DADP-combined UDFs. 

## Usage

1. Before we start, if you are only testing non-UDF parts, you could remove
the ```pycopg2``` and ```sqlalchemy``` in the ```requirements.txt```. Moreover, when you are testing the UDF part in a Linux system like ```Ubuntu```, 
you should check whether ```python-psycopg2``` and ```libpq-dev``` is installed, or you should use ```apt``` to get them before you solve the requirements.

2. Solve the dependencies with ```requirements.txt```. Note that we only support Python 3. 

```
$ pip install -r requirements.txt
```
3. Export the ```src``` file to path. For example, in Windows, you may use:
```
$Env:PYTHONPATH += ";X:\DADP\src"
```
4. Run the mechanism under ```mechanisms``` for initialization phase, then run the corresponding mechanism under ```da-mechanisms```.

## Utility usage
* ```\da-mechanisms\werror.py``` - Calculate the workload error for a given preferred attributes file ```prefer.csv```.
