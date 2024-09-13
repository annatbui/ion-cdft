# ion-cdft

Code to implement LCW-cDFT for solvation

## About the code

This repository contains code to train and perform classical density functional theory (cDFT) for ionic fluids.

## Citation

Please find the associated paper with the code:

***A. T. Bui, S. J. Cox, "Learning classical density functionals for ionic fluids", (2024)***

## Contents
* `data`: Simulation data of density profiles of the mimic SR system for training.
* `training`: Code for training neural networks of the one-body direct correlation functions.
* `models`: Keras models obtained from training.
* `cdft`: Performing cDFT calculation for structure and thermodynamics of the mimic SR and the full LR systems.

## Description



## Installation

You can clone the repository with:
```sh
git clone https://github.com/annatbui/ion-cdft.git
```

For training and evaluating the model, TensorFlow/Keras is used preferably with a GPU.
To create a conda environment containing the required packages 

```sh
conda env create -f environment.yml
```


## License

This code is licensed under the GNU License - see the LICENSE file for details.




