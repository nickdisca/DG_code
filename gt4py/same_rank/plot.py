import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Function to plot solution
def plot_solution(u,x_c,y_c,r,d1,d2,neq,hx,hy, plot_type='contour'):
    x_u    = np.zeros(d1*r)
    y_u    = np.zeros(d2*r)
    unif   = np.linspace(-1,1,r)
    for i in range(d1) :
        x_u[i*r:(i+1)*r] = x_c[i]+hx*unif/2
    for j in range(d2) :
        y_u[j*r:(j+1)*r] = y_c[j]+hy*unif/2

    Y, X = np.meshgrid(y_u, x_u)  # Transposed to visualize properly
    Z    = np.zeros((d1*r,d2*r))
    
    for k in range(neq):
        for j in range(d2):
            for i in range(d1):
                Z[i*r:(i+1)*r,j*r:(j+1)*r] = u[i,j,0,:].reshape(r,r)
    Z[np.abs(Z) < np.amax(Z)/1000.0] = 0.0   # Clip all values less than 1/1000 of peak
                
    if plot_type == 'contour':
        fig, ax = plt.subplots()
        CS = ax.contourf(X, Y, Z)
        fig.colorbar(CS)
    elif plot_type == 'surf':
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # CS = ax.plot_surface(X, Y, Z)
        # fig, ax = plt.subplots()
        CS = ax.plot_surface(X, Y, Z)
    else:
        print("Plot type not recognised!")
    plt.show()
    print("plotted")