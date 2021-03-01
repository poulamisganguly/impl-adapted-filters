[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/poulamisganguly/impl-adapted-filters/HEAD)

# Implementation-adapted filters

This repository has example code for the paper:
Improving reproducibility in synchrotron tomography by implementation-adapted filters by Ganguly et al.

Different tomographic software packages use different discretisations of the reconstruction problem. This results in quantitative differences between the filtered backprojection (FBP) and Fourier-space reconstructions obtained from such packages. In this paper, we propose to optimise the filter used in these algorithms to make reconstructions more similar (and therefore comparable) to each other. 


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/poulamisganguly/impl-adapted-filters.git


## Dependencies

The required dependencies for this project are specified in the file `environment.yml`. We used `conda` for package management.


## Usage

The easiest way to try out our code is by running the `example.ipynb` notebook on Binder.

If you'd like to run the code on your machine, start by creating a `conda` virual environment. 
This can be done by running the following command in the repository folder (where `environment.yml`
is located):

    conda env create -f environment.yml
    
Before running any code you must activate the conda environment:

    conda activate impl-filters

Install JupyterLab
```
    conda install -c conda-forge jupyterlab
```
Start a JupyterLab instance by running

    jupyter lab --no-browser --port=5678 --ip=127.0.0.1

Open the notebook on your web browser by clicking on the generated link or navigating to http://127.0.0.1:5678/

## License

All source code is made available under an MIT license.

## Issues and Contributions

If you have any trouble running the code, please open an issue here or contact me at poulami[at]cwi[dot]nl. Thank you for your interest!

## Acknowledgements
This readme is a modified version of https://github.com/pinga-lab/paper-template/blob/master/README.md
