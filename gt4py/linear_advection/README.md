# Problem

This directory contains files related to our DG solver for the planar linear advection problem with constant velocity field $\beta$:

$$
\frac{\partial u}{\partial t} + \nabla \cdot (\beta u) = 0 \qquad \text{with} \qquad \beta = [1, 1]^T
$$

# Demo

The following figure illustrates the numerical solution of the linear advection problem using our 4th order DG scheme.
The plot depicts a 1 second simulation on the unit sphere with periodic boundary conditions and a cosine bell as initial conditions.

![Linear Advection NUmerical solu](https://user-images.githubusercontent.com/58524567/183125781-918d5460-6e8d-4df6-98d4-d533c751e029.gif)



# Utilisation
The simulation an be executed from the command line using:
```
python main.py 20 3 4 numpy
```
- 20 being the number of elements used for both x and y directions
- 3 being the polynomial degree (this leads to a 4th order scheme in space)
- 4 being the order of Runge-Kutta method
- numpy being the backend
