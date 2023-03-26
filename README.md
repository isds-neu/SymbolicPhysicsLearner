![logo](spl-logo.png)

# SymbolicPhysicsLearner

source code and examples for Symbolic physics learner: Discovering governing equations via Monte Carlo tree search. 

## Dependencies

`requirements.txt`

### Baseline Models
The following packages are not included in the requirements list. Please install at your interest. 

1. base GP model: `pip install gplearn` https://gplearn.readthedocs.io/en/stable/
2. SINDy: `pip install pysindy` https://pysindy.readthedocs.io/en/latest/
3. Deep Symbolic Regression: refer to [dso](https://github.com/brendenpetersen/deep-symbolic-optimization)

## Datasets
### symbolic regression data
Go to  `SymbolicPhysicsLearner/regression_task` and run
```
python makedatasets.py --task=nguyen-1
```
### dropping ball experiment data 
source data is from https://github.com/briandesilva/discovery-of-physics-from-data

To proces the data into the input of SPL model, go to  `SymbolicPhysicsLearner/physics_task` and run
```
python make_datasets_balldrop.py
```
### Lorenz dataset
Lorenz experimental dataset is simulted by MATLAB `ode113` function. 

### double pendulum dataset
Double Pendulum dataset is from camera-recorded experiments provided by https://developer.ibm.com/exchanges/data/all/double-pendulum-chaotic/

Check more details about its pre-processing from `SymbolicPhysicsLearner/dynamics_task/dp_makedata.ipynb`



## Citing the paper (temporary) 
```
@article{sun2022symbolic,
  title={Symbolic physics learner: Discovering governing equations via Monte Carlo tree search},
  author={Sun, Fangzheng and Liu, Yang and Wang, Jian-Xun and Sun, Hao},
  journal={arXiv preprint arXiv:2205.13134},
  year={2022}
}
```
