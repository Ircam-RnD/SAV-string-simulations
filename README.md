# String simulations using Scalar auxiliary method with added control term

This repository contains implementations of the numerical method presented in a [publication](https://hal.science/hal-05222856/document) presented at DAFx2025. This work got awarded the best paper prize at the conference and an extended version is under review for publication in JAES.  
The simulated system is a cubic nonlinear string described by the following PDE:
$\partial_t^2 u = \frac{1}{\mu} [(T \partial_x^2 - EI\partial_x^4) u - \mu (2 \eta_0 - 2 \eta_1\partial_x^2) \partial_t u + \frac{EA - T}{2}\partial_x \left(\partial_x u \right)^3 + F_{ext}],$
with $u(x,t)$ the transversal displacement of the string. Fixed boundary conditions are considered. 

The main contribution of this work is the addition of a control term to the standard SAV leapfrog scheme, reducing the numerical drift of the scalar auxiliary variable. As a result, the method is suitable for real-time sound synthesis without parasitic pitch drifts appearing.

## Structure

**/doc** contains files necessary for the creation of the associated github page. A JS implementation of the model is present and used for the intercative simulation in the browser.

**/python** contains a python implementation and scripts allowing to reproduce figures of the paper.

**/src** contains a standalone C++ implementation and the respective Max/MSP external wrapper.

Details on each implementations is given in the subdirectories.