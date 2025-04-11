# ion-cdft


## About the code

<p align="center">
  <img src="https://github.com/user-attachments/assets/a3756a3f-a6cf-43a0-9a4d-d7aabd7aba93" width="50%">
</p>


This repository contains code to train and perform classical density functional theory (cDFT) for ionic fluids.

## Citation

Please find the associated paper with the code:


***A. T. Bui, S. J. Cox, **"Learning classical density functionals for ionic fluids"**, Phys. Rev. Lett. **134**, 148001 (2025)***

Links to: [arXiv:2410.02556](
https://doi.org/10.48550/arXiv.2410.02556) | [Phys. Rev. Lett.](https://doi.org/10.1103/PhysRevLett.134.148001)


## Contents
* `data`: Simulation data of density profiles of the mimic SR system for training.
* `training`: Code for training neural networks of the one-body direct correlation functions.
* `models`: Keras models obtained from training.
* `cdft`: Performing cDFT calculation for structure and thermodynamics of the mimic SR and the full LR systems.


## Training data

Raw training data of the ML models are deposited on [Zenodo](https://zenodo.org/records/15085645).


## Installation

You can clone the repository with:
```sh
git clone https://github.com/annatbui/ion-cdft.git
```

For training and evaluating the model, *TensorFlow/Keras* is used. Performance is better with a GPU.
To create a *conda* environment containing the required packages 

```sh
conda env create -f environment.yml
```



## License

This code is licensed under the GNU License - see the LICENSE file for details.




