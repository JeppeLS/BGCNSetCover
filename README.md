# Solving the Set Covering Problem with a Bipartite Graph Convolutional Network

## Contents

* [Introduction](https://github.com/JeppeLS/speciale#Introduction)
* [Installation](https://github.com/JeppeLS/speciale#Installation)
* [Replicating Results](https://github.com/JeppeLS/speciale#Replicating-results)

## Introduction
This repository contains the code for the thesis "Learning Heuristics for the Unicost Set Covering Problem" by Jeppe Liborius Sjørup and Kasper Urup Reimer.

## Installation
To test the code, first clone the repository
```
https://github.com/JeppeLS/speciale.git
```
Next install dependencies by running:
```
pip install -r requirements.txt
```
Furthermore a working installation of Pytorch Geometric is needed. Installations instructions can be found at https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

## Replicating results

The tests we conducted in this thesis are included as methods in the "run_network_test.py" file. 
As standard if the file is run then the GCN-based model will be benchmarked on all of the instances in OR-library.
Other models can be benchmarked, by changing the network type parameter in the call to the beasley test method.

For conducting other tests, edit the 'run_network_test.py' file by uncommenting the tests you want to perform.

The BGCN model is implemented in the model subfolder, and the main class responsible for evaluating and training the models
can be found in 'main.py'. 
