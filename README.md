# Numerical Luminosity Inversion

## Repository in order to compute the transverse emittances from the luminosity

**Authors:** M. Rufolo

**Requirements:** Python 3.12

## To install:
```bash
git clone https://github.com/mattrufolo/NumericalLumiInversion.git
pip install ./NumericalLumiInversion
cd NumericalLumiInversion
pip install ./inv_gauss_tree_maker
```

## To run some examples:
```bash
cd examples
python scanner_penalty12.py
python scanner_penaltyxy.py
python scanner_penaltyxy_err.py
```

From this three examples it is possible to achieve the 3 different plots represented in the paper. Obiovously the user can decide how sharp should be the
grid for the different scanner.

## Usage

### Numerical Inversion

Given the machine parameters, the different scripts in "inversion_fun" use a perturbative method in order to obtain from the non-linear Least Squares of scipy the transverse emittances from the luminosity model. For example the user can run the following example

```python
from inversion_fun import Inv_gauss_xy as  inv_g

import random

#random choose of emittance
epsx = random.uniform(0.9*2.3e-6,1.1*2.3e-6)
epsy = random.uniform(0.9*2.3e-6,1.1*2.3e-6)

#perturbation on offset
dict_shift = {'mu0x':[0.01]}

#numerical inversion
[sol,f,J,time,iter] = inv_g.inv_gauss_xy(epsx,epsy,dict_shift)
```

The output of this computation will be:
- sol: the numerical solution
- f: the function of the system at the numerical solution
- J: the Jacobian of the system at the numerical solution
- time: the computation time
- iter: the number of iteration of the Least Squares.

As choice for the different parameters, it has be choosen a default choice, inspiring on the LHC parameters:

Parameters | value 
--- | --- 
dθ<sub>x</sub>,dθ<sub>y</sub>,dμ<sub>x</sub>,dμ<sub>y</sub> | 0  
Δθ<sub>0x</sub> | 0 μrad
Δθ<sub>0y</sub> | 320 μrad
 ,β<sub>x2</sub><sup>*</sup> ,β<sub>y1</sub><sup>*</sup> ,β<sub>y2</sub><sup>*</sup> | 30 cmX<sub>x1</sub><sup>*</sup>
α<sub>x1</sub>,α<sub>x2</sub>,α<sub>y1</sub>,α<sub>y2</sub>| 0 cm 
σ<sub>z1</sub>,σ<sub>z2</sub>| 9 cm


Anyway the user can be changed it.

### Parallel computation

Some results in a subfolder of the examples have been done with the help of a tree maker, in order to simulate more choices of the beam transverse emittances.

Running in "inv_gauss_tree_maker"

```bash
python 003_post_process
```

It is possible to obtain different plots in the subfolder trees of examples/. In particular for each case analyzed, it has been represented:

- in one_eps subfolder: the different iterations of the LS before reaching the its final guess
- the plots of the different choices of the emittances and all their guesses
- the histogram of the relative error for each component of the estimation.


## Last example

Unfortunately for a memory reason, I could not represent the most interesting and final result of the case with all the different emittances between axes and beams, even considering the luminosity measurement error. If the user is interested in doing that he can use ~20 CPUs to see the result himself. He has to run in the inv_gauss_tree_maker subfolder

```bash
python 001_make_folders.py
python 002_cronjob.py
```
Once the tree_maker.log in the root is 'completed'. You can run 

```bash
python 002_cronjob.py
```

Once also all the 15 leafs are 'completed' in the tree_maker.log, it is possible to see the result running

```bash
python 003_last_post_process.py
```

All these computation should last ~1day, it is possible to speed up the computation using more threads, and reducing the number of iteration for each thread in the config.yaml in the folder inv_gauss_tree_maker.

## Acknowledgements

All the computations done in this repository has been done in parallel, using the tree maker structure from the [gitlab_repository](https://gitlab.cern.ch/abpcomputing/sandbox/tree_maker.git)
