# Problem
This directory contains files related to our DG solver for the 2D planar Shallow Water Equations:

$$
\begin{cases}
    \partial_t (h) + \partial_x (hu) + \partial_y (hv) = 0 \\
    \partial_t (hu) + \partial_x (hu^2 + \frac{1}{2}gh^2) + \partial_y (huv) = 0 \\
     \partial_t (hu) + \partial_x (huv) = 0  + \partial_y (hv^2 + \frac{1}{2}gh^2) = 0\\
\end{cases}   
$$

# Demo
The following figure illustrates the numerical solution for the planar Shallow Water Equations using our 4th order DG scheme.
The plot depicts the water height component evolved on a square domain with periodic boundary conditions.
![Shallow Water Equations](https://user-images.githubusercontent.com/58524567/183141418-8cd5be6e-aaff-4640-9097-de5c85f6ca86.gif)

# Utilisation
The simulation an be executed from the command line using:
```sh
python main.py 20 1 3 4 numpy
```
- 20 being the number of elements used for both x and y directions
- 1 being number of identical vertical levels
- 3 being the polynomial degree (this leads to a 4th order scheme in space)
- 4 being the order of Runge-Kutta method
- numpy being the backend
