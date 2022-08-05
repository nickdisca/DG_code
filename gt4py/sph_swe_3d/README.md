# Problem

This directory contains the source code for the DG solver for the Shallow Water Equations on the sphere:

$$
    \begin{cases}
        \partial_t (h \cos\theta)  + \cfrac{1}{R}\left[ \partial_\lambda \left(h u\right) + \partial_\theta \left(h v \cos\theta\right) \right] = 0 \\
        \partial_t (h u \cos\theta)  +  \cfrac{1}{R}\left[\partial_\lambda \left(h u^2 + \cfrac{g h^2}{2}\right) + \partial_\theta \left(h u v \cos\theta\right) \right] = f h v \cos\theta \\ 
        \partial_t (h v \cos\theta) + \cfrac{1}{R}\left[ \partial_\lambda \left(h u v\right) + \partial_\theta \left(\left(h v^2 + \cfrac{g h^2}{2}\right) \cos\theta\right) \right] = \cfrac{g h^2 \sin \theta}{2 R} - f h u \cos\theta
    \end{cases}
$$

![Rossby-Haurwitz wave](https://user-images.githubusercontent.com/58524567/183117994-13e4c36b-0ffe-4a3f-8241-4acef8ed4859.gif)



# Utilisation
The simulation an be executed from the command line using:
```
python main.py 20 1 3 4 numpy
```
- 20 being the number of longitudinal elements (half this number will be chosen for the number of latitudinal elements)
- 1 being number of identical vertical levels
- 3 being the polynomial degree (this leads to a 4th order scheme in space)
- 4 being the order of Runge-Kutta method
- numpy being the backend
