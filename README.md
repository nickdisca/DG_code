# DGMG
### Created by: Niccolò Discacciati (EPFL, Lausanne), Forked and Updated by Will Sawyer (CSCS/ETH)
### Code available at: 
`https://github.com/vectorflux/DG_code/`

### Date: 16 March, 2020

**Latest update**  On the "devel" branch there is now a refactored version of Nick's code which uses a simple {i,j,n} structure over all arrays (i over the X and j over the Y dimension, n over the number of equations).   Besides being easier to read than the old verion, this one has a structure which is compatible with the GridTools library, and therefore might serve as a basis to port the algorithm to Python, possibly using the GT4Py extension.

This version is run out of **driver_new.m**, which has the same parameters as the origin **driver.m** code described below.  The revised functions generally have the suffix **new**, although be aware that not all the functions with that suffix are actually used.   The full implementation of the original code is still present in the **devel** branch and essentially unchanged.  This allows for direct comparisons between the two versions for debugging. 

### Overview


**DGMG** is a modal RKDG-solver written in MATLAB, which can handle conservation laws in both cartesian and spherical coordinates (MG stands for 'MultiGeometry'), with a support for variable degree in space ('static adaptivity'). Details about DG discretization methods can be found in, e.g.

 - J. S. Hesthaven, T. Warburton. "Nodal Disontinuous Galerkin methods." Springer, 2008. 

A more detailed description is provided in, e.g.,

 - S. Carcano. “Finite volume methods and Discontinuous Galerkin methods for the numerical modeling of multiphase gas-particle flows.” Doctoral Thesis. Politecnico di Milano, 2014.
- G. Tumolo, L. Bonaventura. “A semi-implicit, semi-Lagrangian Discontinuous Galerkin framework for adaptive numerical weather prediction.” Q.J. Royal Meteorological Society, 2015.
- G. Tumolo, L. Bonaventura, M. Restelli. “A semi-implicit, semi-Lagrangian, p-adaptive dis- continuous Galerkin method for the shallow water equations.” Journal of Computational Physics (2013).

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
r_max=2; 

%degree distribution
r=degree_distribution("y_dep",d1,d2,r_max);

%equation type
eq_type="linear";

%type of quadrature rule
quad_type="leg";

%number of quadrature points in one dimension
n_qp_1D=8;

%time interval, initial and final time
t=0;
T=1;

%order of the RK scheme
RK=3; 

%time step
dt=1e-4;

%plotting frequency
plot_freq=100;

%initial condition
u0_fun=@(x,y) sin(2*pi*x).*sin(2*pi*y);

~~~

* The numerical domain is set as as a rectangle `[a,b]x[c,d]`.

* The number of elements in the horizontal and vertical direction is set by `d1` and `d2` respectively. This results in a structured grid, made of `d1*d2` rectangles.

* The maximum spatial order of the method is set by `r_max`, while the spatial distribution of the degrees is handled by the routine `degree_distribution.m`. Available options are
 * `"unif"` which sets a constant degree in all elements, equal to the maximum order allowed.
 * `"y_dep"` which sets a vertically variable degree, which is maximum at the half of the domain and symmetrically decreases to `1` towards the boundaries. 
 * `"y_incr"` which sets a monotonically increasing degree in the vertical direction.

	Within each element, the degrees of freedom are the modal coefficients of the solution, whereas the basis consists of the tensor product of one-dimensional Legendre polynomials.
	
* The quadrature rule is specified by the flag `quad_type`, to be set to `"leg"` or `"lob"` to use the Gauss-Legendre or Gauss-Legendre-Lobatto points, respectively. The number of one-dimensional quadrature points is set by `n_qp_1D`, and it should be at least equal to `r_max+1` and `r_max+2`, repsectively, to be able to integrate exactly polynomials of order `2*r_max` (e.g., mass matrix).

* The time interval is specified by `[t,T]`, with a time-step `dt`. The Runge-Kutta order is set by `RK`.
* The initial condition is specified using the function `u0_fun`.
* Several equation are available 
 * `"linear"`: Linear advection equation in cartesian geometry with the advection speed set to `[1 1]`.
 * `"swe"` : Shallow water equations in cartesian geometry.
 * `"adv_sphere"`: Spherical linear advection equation. The space-dependent advection field is parametrized by an angle `alpha`. To modify its value, the files `flux_function.m` and `get_maximum_eig.m` have to be changed.
 * `"swe_sphere"`: Shallow water equations in spherical geometry.

* The variable `plot_freq` determines how often the solution should be visualized.

* Once the program terminates, the modal solution values can be accessed from the variable `u`. A plot of all conserved variables is also shown.

## Known issues
* Even though the qualitative behavior in spherical coordinates is correct, the convergence analysis seems to give a sub-optimal order.
* In order to support static adaptivity, the computational time is quite large. Even though minor optimizations could be done, we believe that to obtain significant speedup one should consider implementations in other languages.
* Dynamic adaptivity is not handled at the moment. Even though the code can easily handle it, the computational performance would drastically drops. This happens because several operations have to be repeated at each time step (e.g., computing the mass matrix), causing a huge overhead.
