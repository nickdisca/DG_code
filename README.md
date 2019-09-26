# DGMG
### Created by: Niccolò Discacciati (EPFL, Lausanne) 
### Code available at: 
`https://github.com/nickdisca/DG_code/`             

### Date: 26 September, 2019

**DGMG** is a modal RKDG-solver written in MATLAB, which can handle conservation laws in both cartesian and spherical coordinates (MG stands for 'MultiGeometry'). Details about DG discretization methods can be found in, e.g.

 - J. S. Hesthaven, T. Warburton. "Nodal Disontinuous Galerkin methods." Springer, 2008. 

A more detailed description is provided in, e.g.,

 - S. Carcano. “Finite volume methods and Discontinuous Galerkin methods for the numerical modeling of multiphase gas-particle flows.” Doctoral Thesis. Politecnico di Milano, 2014.
- G. Tumolo, L. Bonaventura. “A semi-implicit, semi-Lagrangian Discontinuous Galerkin framework for adaptive numerical weather prediction.” Q.J. Royal Meteorological Society, 2015.- G. Tumolo, L. Bonaventura, M. Restelli. “A semi-implicit, semi-Lagrangian, p-adaptive dis- continuous Galerkin method for the shallow water equations.” Journal of Computational Physics (2013).

Interesting test cases are provided in 

- D. Williamson et al. “A standard test for numerical approximation to the shallow water equations in spherical geometry.” Journal of Computational Physics (1992).


## Running the code 
To execute the code, it is enough to run the script `driver.m`. A few parameters have to defined as follows:

~~~matlab
%definition of the domain [a,b]x[c,d]
a=0; b=1; c=0; d=1;

%number of elements in x and y direction
d1=40; d2=40; 

%polynomial degree of DG
r=1; 

%equation type
eq_type="linear";

%time interval, initial and final time
t=0;
T=1;

%order of the RK scheme
RK=3; 

%time step
dt=1e-4;

%initial condition
u0_fun=@(x,y) sin(2*pi*x).*sin(2*pi*y);

~~~

* The numerical domain is set as as a rectangle `[a,b]x[c,d]`.
* The number of elements in the horizontal and vertical direction is set by `d1` and `d2` respectively. This results in a structured grid, made of `d1*d2` rectangles.
* The spatial order of the method is set by `r`.
* The time interval is specified by `[t,T]`, with a time-step `dt`. The Runge-Kutta order is set by `RK`.
* The initial condition is specified using the function `u0_fun`.
* Several equation are available 
 * `"linear"`: Linear advection equation in cartesian geometry with the advection speed set to `[1 1]`.
 * `"swe"` : Shallow water equations in cartesian geometry.
 * `"adv_sphere"`: Spherical linear advection equation. The space-dependent advection field is parametrized by an angle `alpha`. To modify its value, the files `flux_function.m` and `get_maximum_eig.m` have to be changed.
 * `"swe_sphere"`: Shallow water equations in spherical geometry.
* Once the program terminates, the modal solution values can be accessed. A plot of all conserved variables is also shown.