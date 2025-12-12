# Python simulation code

## Structure

**`StringGeom.py`** contains an implementation of the controlled SAV algorithm and of the semi-implicit integrator for cubic nonlinearities, both applied to the cubic nonlinear string. 

**`run_convergence.py`** is a python script running and plotting convergence analysis with samplerate, either at fixed spatial discretisation or at constant CFL ratio. 

**`lambda_effect.py`** is a python script running the same simulations with different values for the control parameter $\lambda_0$.

**`config.json`** is a configuration file for the numerical experiments. It defines physical constants, initial conditions, result folder and convergence settings. 

## Dependencies

Dependencies are listed in **`requirements.txt`**. In a python virtual environment, you can install the dependencies using the following command:
```pip install -r requirements.txt```.