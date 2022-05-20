from dataclasses import dataclass
import numpy as np
import quadpy as qp
import math
import sys
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Radius of the earth (for spherical geometry)
# radius=6.37122e6
radius = 1

# Equation type
# eq_type="linear";
# eq_type="adv_sphere";
eq_type="swe";

# number of elements in X and Y
d1=20; d2=20


# definition of the domain [a,b]x[c,d]
# a=0; b=2*np.pi; c=-np.pi/2; d=np.pi/2;
# a=0; b=1; c=0; d=1
a=0; b=1e7; c=0; d=1e7

# length of the 1D intervals
hx=(b-a)/d1; hy=(d-c)/d2

# polynomial degree of DG
r_max=1

# cardinality
dim=(r_max+1)**2

# Type of quadrature rule (Gauss-Legendre or Gauss-Legendre-Lobatto)
quad_type="leg"

# Number of quadrature points in one dimension
n_qp_1D=2

# Number of quadrature points
n_qp=n_qp_1D*n_qp_1D

# Time interval, initial and final time
t=0
# T=1000
#T=86400
#T=5*86400
T = 1 * (b-a)

# Order of the RK scheme (1,2,3,4)
RK=1

# Time step
# For "linadv":  dt=1/r_max^2*min(hx,hy)*0.1;
# For "adv_sphere" with earth radius
# dt = 1e-3

# Plotting frequency (time steps)
plot_freq=50

# Derived temporal loop parameters
# Courant=dt/min(hx,hy)
Courant = 0.0001

dx =min(hx,hy)
dt = Courant * dx / (r_max+1)



N_it=math.ceil(T/dt)

# Coriolis function currently zero
coriolis_fun=lambda x,y: np.zeros(len(x));  # Needed only for swe_sphere

# Function to plot solution
def plot_solution(u,x_c,y_c,r,d1,d2,neq,hx,hy):
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
                Z[i*r:(i+1)*r,j*r:(j+1)*r] = u[i,j,0].reshape(r,r)
    # Z[np.abs(Z) < np.amax(Z)/1000.0] = 0.0   # Clip all values less than 1/1000 of peak
                
    fig, ax = plt.subplots()
    CS = ax.contourf(X, Y, Z)
    plt.colorbar(CS)
    plt.pause(0.05)

# Function to compute the element degrees
def degree_distribution(type,d1,d2,r_max):
    if type == "unif":
        rdist = r_max*np.ones((d1, d2))
    elif type == "y_dep":
# Implement:  round( (r_max-1)/(floor(d2/2)-1)*(0:floor(d2/2)-1)+1 );
        rdist = r_max*np.ones((d1, d2))
    elif type == "y_incr":
# Implement: round ( (r_max-1)/(d2-1)*(0:d2-1)+1 );
        rdist = r_max*np.ones((d1, d2))
    else: 
        rdist = r_max*np.ones((d1, d2))
        print (type,"unsupported degree distribution, using uniform")
    rdist.astype(int)
    return rdist

# Function to determine the Rusanov flux
def flux_function(eq_type,d1,d2,u,radius,hx,hy,x_c,y_c,pts_x,pts_y):
    flux_x = {}
    flux_y = {}

    if eq_type == "linear" :
        meters_per_s = 1
        for j in range(d2):
            for i in range(d1):
                flux_x[i,j,0] = meters_per_s * u[i,j,0]
                flux_y[i,j,0] = meters_per_s * u[i,j,0]

    elif eq_type == "swe" :
#
# Currently untested; in original version g = 0, not sure why
#
        g=9.80616;
        for j in range(d2):
            for i in range(d1):
                flux_x[i,j,0] = u[i,j,1]
                flux_y[i,j,0] = u[i,j,2]
                flux_x[i,j,1] = np.square(u[i,j,1])/u[i,j,0] + g/2*np.square(u[i,j,0])
                flux_y[i,j,1] = u[i,j,1] * u[i,j,2] / u[i,j,0]
                flux_x[i,j,2] = u[i,j,1] * u[i,j,2] / u[i,j,0]
                flux_y[i,j,2] = np.square(u[i,j,2])/u[i,j,0] + g/2*np.square(u[i,j,0])

    elif eq_type == "adv_sphere" :
        angle = np.pi / 2
        meters_per_s = 2*np.pi*radius/(12*86400)
        flux_x = {}
        flux_y = {}
        for j in range(d2):
            for i in range(d1):
                qp_x = x_c[i] + pts_x*hx/2
                qp_y = y_c[j] + pts_y*hy/2
                beta_x = meters_per_s * (np.cos(qp_y)*np.cos(angle) + np.sin(qp_y) * np.cos(qp_x)*np.sin(angle))
                beta_y = -meters_per_s * (np.sin(angle)*np.sin(qp_x))
                flux_x[i,j,0] = beta_x * u[i,j,0]
                flux_y[i,j,0] = beta_y * u[i,j,0]

    elif eq_type == "swe_sphere" :
#
# Currently untested; original version is identical to "swe", but this cannot be correct
#
        g=9.80616;
        for j in range(d2):
            for i in range(d1):
                flux_x[i,j,0] = u[i,j,1]
                flux_y[i,j,0] = u[i,j,2]
                flux_x[i,j,1] = np.square(u[i,j,1])/u[i,j,0] + g/2*np.square(u[i,j,0])
                flux_y[i,j,1] = u[i,j,1] * u[i,j,2] / u[i,j,0]
                flux_x[i,j,2] = u[i,j,1] * u[i,j,2] / u[i,j,0]
                flux_y[i,j,2] = np.square(u[i,j,2])/u[i,j,0] + g/2*np.square(u[i,j,0])

    else :
        print (eq_type,"unsupported option, using linear ")
        flux_x = {}
        flux_y = {}
        for j in range(d2):
            for i in range(d1):
                flux_x[i,j,0] = u[i,j,0]
                flux_y[i,j,0] = u[i,j,0]

    return flux_x, flux_y

# Function to compute the fluxes into or out of the DG elements
def comp_flux_bd(d1,d2,neq,u_n,u_s,u_e,u_w,pts_x,pts_y,hx,hy,eq_type,radius,x_c,y_c):
    
    # Find the U values from the N, S, E, W neighboring cells

    fx_n, fy_n = flux_function(eq_type,d1,d2,u_n,radius,hx,hy,x_c,y_c+hy/2,pts_x,np.zeros(len(pts_x)))
    fx_s, fy_s = flux_function(eq_type,d1,d2,u_s,radius,hx,hy,x_c,y_c-hy/2,pts_x,np.zeros(len(pts_x)))
    fx_e, fy_e = flux_function(eq_type,d1,d2,u_e,radius,hx,hy,x_c+hx/2,y_c,np.zeros(len(pts_y)),pts_y)
    fx_w, fy_w = flux_function(eq_type,d1,d2,u_w,radius,hx,hy,x_c-hx/2,y_c,np.zeros(len(pts_y)),pts_y)

    flux_n = {}
    flux_s = {}
    flux_e = {}
    flux_w = {}

    if eq_type == "linear" :

        alpha = 1.0
        for j in range(d2):
            for i in range(d1):

                i_n=i; j_n=(j+1)%d2;     # Find the index of neighbor sharing north edge 
                i_s=i; j_s=(j-1)%d2;     # Find the index of neighbor sharing south edge 
                i_e=(i+1)%d1; j_e=j;      # Find the index of neighbor sharing east edge 
                i_w=(i-1)%d1; j_w=j;      # Find the index of neighbor sharing west edge 

                flux_n[i,j,0] =  1/2*(fy_n[i,j,0]+fy_s[i_n,j_n,0]) - alpha/2 * (u_s[i_n,j_n,0] - u_n[i,j,0])
                flux_s[i,j,0] = -1/2*(fy_s[i,j,0]+fy_n[i_s,j_s,0]) - alpha/2 * (u_n[i_s,j_s,0] - u_s[i,j,0])
                flux_e[i,j,0] =  1/2*(fx_e[i,j,0]+fx_w[i_e,j_e,0]) - alpha/2 * (u_w[i_e,j_e,0] - u_e[i,j,0])
                flux_w[i,j,0] = -1/2*(fx_w[i,j,0]+fx_e[i_w,j_w,0]) - alpha/2 * (u_e[i_w,j_w,0] - u_w[i,j,0])

    elif eq_type == "adv_sphere" :

        angle     = np.pi/2
        cos_angle = 0.0
        sin_angle = 1.0
        meters_per_s = 2*np.pi*radius/(12*86400)   # Over twelve days

        for j in range(d2):
            qp_y   = y_c[j]+pts_y*hy/2;         # Vector
            qp_y_n = y_c[j]+hy/2; qp_y_s = y_c[j]-hy/2;   # Scalars


            if ( j==d2 ) :
                fact_bd_n = 0.0          # Avoid epsilon values in north pole
            else :
                fact_bd_n = np.cos(qp_y_n)  # Scalar

            if ( j==1 ) :
                fact_bd_s = 0.0
            else :
                fact_bd_s = np.cos(qp_y_s)  # Scalar

            for i in range(d1):
                qp_x   = x_c[i]+pts_x*hx/2;
                qp_x_e = x_c[i]+hx/2; qp_x_w = x_c[i]-hx/2; 
                # The beta_x_* are all vectors
                beta_x_n = meters_per_s*(np.cos(qp_y_n)*cos_angle+np.sin(qp_y_n)*np.cos(qp_x)*sin_angle)
                beta_x_s = meters_per_s*(np.cos(qp_y_s)*cos_angle+np.sin(qp_y_s)*np.cos(qp_x)*sin_angle)
                beta_x_e = meters_per_s*(np.cos(qp_y)*cos_angle+np.sin(qp_y)*np.cos(qp_x_e)*sin_angle)
                beta_x_w = meters_per_s*(np.cos(qp_y)*cos_angle+np.sin(qp_y)*np.cos(qp_x_w)*sin_angle)
                beta_y   =-meters_per_s*sin_angle*np.sin(qp_x);     # Vector
                beta_y_e =-meters_per_s*sin_angle*np.sin(qp_x_e);   # Scalar
                beta_y_w =-meters_per_s*sin_angle*np.sin(qp_x_w);   # Scalar

                alpha_n = np.amax( np.sqrt(np.square(beta_x_n)+np.square(beta_y)), axis=0 ); 
                alpha_s = np.amax( np.sqrt(np.square(beta_x_s)+np.square(beta_y)), axis=0 );
                alpha_e = np.amax( np.sqrt(np.square(beta_x_e)+beta_y_e**2), axis=0 ); 
                alpha_w = np.amax( np.sqrt(np.square(beta_x_w)+beta_y_w**2), axis=0 );

                i_n=i; j_n=(j+1)%d2;     # Find the index of neighbor sharing north edge
                i_s=i; j_s=(j-1)%d2;     # Find the index of neighbor sharing south edge
                i_e=(i+1)%d1; j_e=j;     # Find the index of neighbor sharing east edge
                i_w=(i-1)%d1; j_w=j;     # Find the index of neighbor sharing west edge

                flux_n[i,j,0] = fact_bd_n*(1/2*(fy_n[i,j,0]+fy_s[i_n,j_n,0]) - alpha_n/2 * (u_s[i_n,j_n,0] - u_n[i,j,0]))
                flux_s[i,j,0] = fact_bd_s*(-1/2*(fy_s[i,j,0]+fy_n[i_s,j_s,0]) - alpha_s/2 * (u_n[i_s,j_s,0] - u_s[i,j,0])) 
                flux_e[i,j,0] =  1/2*(fx_e[i,j,0]+fx_w[i_e,j_e,0]) - alpha_e/2 * (u_w[i_e,j_e,0] - u_e[i,j,0])
                flux_w[i,j,0] = -1/2*(fx_w[i,j,0]+fx_e[i_w,j_w,0]) - alpha_w/2 * (u_e[i_w,j_w,0] - u_w[i,j,0])

    elif eq_type == "swe" :

        g = 9.80616
        fun_alpha = lambda g,u1,u2,u3 : np.amax( np.sqrt(np.abs(g*u1))+np.sqrt(np.square(u2/u1)+np.square(u3/u1)) )
        courant = 0.0001
        alpha_const = courant * dx / dt
        for j in range(d2):
            for i in range(d1):
                i_n=i; j_n=(j+1)%d2;     # Find the index of neighbor sharing north edge
                i_s=i; j_s=(j-1)%d2;     # Find the index of neighbor sharing south edge
                i_e=(i+1)%d1; j_e=j;     # Find the index of neighbor sharing east edge
                i_w=(i-1)%d1; j_w=j;     # Find the index of neighbor sharing west edge

                # Calculate the maximum wave speed over the north face (same for all QP)
                alpha = max( fun_alpha(g,u_n[i,j,0],u_n[i,j,1],u_n[i,j,2]), 
                             fun_alpha(g,u_s[i_n,j_n,0],u_s[i_n,j_n,1],u_s[i_n,j_n,2]))
                for n in range(neq):
                    flux_n[i,j,n] = 1/2*(fy_n[i,j,n]+fy_s[i_n,j_n,n]) - alpha_const/2 * (u_s[i_n,j_n,n] - u_n[i,j,n])

                # Calculate the maximum wave speed over the south face (same for all QP)
                alpha = max( fun_alpha(g,u_s[i,j,0],u_s[i,j,1],u_s[i,j,2]), 
                             fun_alpha(g,u_n[i_s,j_s,0],u_n[i_s,j_s,1],u_n[i_s,j_s,2]))
                for n in range(neq):
                    flux_s[i,j,n] = 1/2*(fy_s[i,j,n]+fy_n[i_s,j_s,n]) - alpha_const/2 * (u_n[i_s,j_s,n] - u_s[i,j,n])

                # Calculate the maximum wave speed over the east face (same for all QP)
                alpha = max( fun_alpha(g,u_e[i,j,0],u_e[i,j,1],u_e[i,j,2]),
                             fun_alpha(g,u_w[i_e,j_e,0],u_w[i_e,j_e,1],u_w[i_e,j_e,2]))
                for n in range(neq):
                    flux_e[i,j,n] = 1/2*(fy_e[i,j,n]+fy_w[i_e,j_e,n]) - alpha_const/2 * (u_w[i_e,j_e,n] - u_e[i,j,n])

                # Calculate the maximum wave speed over the west face (same for all QP)
                alpha = max( fun_alpha(g,u_w[i,j,0],u_w[i,j,1],u_w[i,j,2]),
                             fun_alpha(g,u_e[i_w,j_w,0],u_e[i_w,j_w,1],u_e[i_w,j_w,2]))
                for n in range(neq):
                    flux_w[i,j,n] = 1/2*(fy_w[i,j,n]+fy_e[i_w,j_w,n]) - alpha_const/2 * (u_e[i_w,j_w,n] - u_w[i,j,n])


    else :
        print( eq_type, " not supported in comp_flux_bd" )

    return flux_n, flux_s, flux_e, flux_w


# Function to compute mass matrix and its inverse
def compute_mass(phi,wts2d,d1,d2,rdist,hx,hy,y_c,pts2d_y,eq_type) :
    determ   = hx*hy/4
    mass     = {}
    inv_mass = {}

    cos_factor = {}
    for j in range(d2):
        if (eq_type == "linear" or eq_type == "swe") :
            cos_factor[j] = np.ones(len(pts2d_y))
        elif (eq_type == "adv_sphere" or eq_type == "swe_sphere") :
            cos_factor[j] = np.cos( y_c[j]+pts2d_y*hy/2 ) 
        else :
            print (eq_type,"unsupported option, using linear ")
            cos_factor[j] = np.ones(d2)

    for j in range(d2):
        for i in range(d1):
            r_loc=rdist[i,j]
            dim = int((r_loc+1)*(r_loc+1))
            matrix = np.zeros((dim,dim))
            for m in range(dim):
                for n in range(dim):
                    # det*sum(i-th basis function in qp * j-th basis function in qp * metric factor * weights)
                    matrix[m,n]=determ * np.dot(wts2d, (phi[r_loc][:,m] * phi[r_loc][:,n] * cos_factor[j]))
            mass[i,j] = matrix
            inv_mass[i,j] = np.linalg.inv(mass[i,j])

    return mass, inv_mass

# Function to compute the right-hand side including the volume integral as the sources due to flux
def compute_rhs(d1, d2, neq, u, rdist, n_qp_1D, phi_val_cell, phi_grad_cell_x, phi_grad_cell_y,
                phi_val_bd_cell_n, phi_val_bd_cell_s, phi_val_bd_cell_e, phi_val_bd_cell_w, inv_mass,
                hx, hy, wts, wts2d, radius, pts_x, pts_y, pts2d_x, pts2d_y, x_c, y_c, coriolis_fun, eq_type) :

    rhsu = {}   # Result

# cardinality and qp
    n_qp=n_qp_1D**2

# determinants of the internal and boundary mappings: 1=bottom, 2=right, 3=top, 4=left
    determ=hx*hy/4;
    bd_det = np.array([ hx/2, hy/2, hx/2, hy/2 ])

# compute solution in the quadrature points (a matrix-vector multiplication)
    u_qp = {}
    for n in range(neq):
        for j in range(d2):
            for i in range(d1):
                u_qp[i,j,n] = np.matmul(phi_val_cell[int(rdist[i,j])],u[i,j,n])

# compute physical value of F(x) inside the region 
    flux_fun_x, flux_fun_y = flux_function(eq_type, d1, d2, u_qp, radius, hx, hy, x_c, y_c, pts2d_x, pts2d_y)

    factors = np.ones(d2)
    if (eq_type == "adv_sphere" or eq_type == "swe_sphere") :
# For spherical geometry adjust the NS deformation
        factors = {}
        for j in range(d2):
            factors[j]=np.cos(y_c[j]+pts2d_y/2*hy)

    u_qp_bd_n = {}
    u_qp_bd_s = {}
    u_qp_bd_e = {}
    u_qp_bd_w = {}
    for n in range(neq):
        for j in range(d2):
            for i in range(d1):
                rhsu[i,j,n] = ( np.matmul( phi_grad_cell_x[rdist[i,j]].T,flux_fun_x[i,j,n]*wts2d ) * (2/hx) * determ +
                              np.matmul( phi_grad_cell_y[rdist[i,j]].T,(factors[j]*flux_fun_y[i,j,n]*wts2d) ) * (2/hy) * determ )

                # Interpolate to the boundaries
                u_qp_bd_n[i,j,n] = np.matmul(phi_val_bd_cell_n[rdist[i,j]],u[i,j,n])
                u_qp_bd_s[i,j,n] = np.matmul(phi_val_bd_cell_s[rdist[i,j]],u[i,j,n])
                u_qp_bd_e[i,j,n] = np.matmul(phi_val_bd_cell_e[rdist[i,j]],u[i,j,n])
                u_qp_bd_w[i,j,n] = np.matmul(phi_val_bd_cell_w[rdist[i,j]],u[i,j,n])

# compute LF fluxes on all four edges : calculate all edges simultaneously
    flux_n, flux_s, flux_e, flux_w = comp_flux_bd( d1, d2, neq, u_qp_bd_n, u_qp_bd_s, u_qp_bd_e, u_qp_bd_w,
                                                   pts_x, pts_y, hx, hy, eq_type, radius, x_c, y_c )

    for n in range(neq):
        for j in range(d2):
            for i in range(d1):
                rhsu[i,j,n] = rhsu[i,j,n] - (0.5*hx*np.matmul(phi_val_bd_cell_n[rdist[i,j]].T,(flux_n[i,j,n]*wts)) +
                                             0.5*hx*np.matmul(phi_val_bd_cell_s[rdist[i,j]].T,(flux_s[i,j,n]*wts)) +
                                             0.5*hy*np.matmul(phi_val_bd_cell_e[rdist[i,j]].T,(flux_e[i,j,n]*wts)) +
                                             0.5*hy*np.matmul(phi_val_bd_cell_w[rdist[i,j]].T,(flux_w[i,j,n]*wts)))
 
    if (eq_type == "swe" or eq_type == "swe_sphere") :
        print( eq_type, " Coriolis not yet supported -- no correction" )

# invert the (local) mass matrix and divide by radius
    for n in range(neq):
        for j in range(d2):
            for i in range(d1):
                rhsu[i,j,n] = 1/radius * np.matmul(inv_mass[i,j],rhsu[i,j,n])

    return rhsu

# Function to interpolate, generally from modal to nodal space
def modal2nodal(d1,d2,neq,uM,V,rdist):
    result = {}
    for n in range(neq):
        for j in range(d2):
            for i in range(d1):
                result[i,j,n] =  np.matmul(V[rdist[i,j]],uM[i,j,n])
    return result

# Function to interpolate, generally from nodal to modal space
def nodal2modal(d1,d2,neq,uM,V,rdist):
    result = {}
    for n in range(neq):
        for j in range(d2):
            for i in range(d1):
                result[i,j,n] =  np.linalg.solve(V[rdist[i,j]],uM[i,j,n])
    return result

# Function to calculate diagonal matrix entries (length r^2) corresponding to normalization
def norm_coeffs(r):
    result = np.zeros(r*r)
    ind = 0
    for k1 in range(r):
        for k2 in range(r):
            result[ind] = math.sqrt((2*k1+1)*(2*k2+1)) / 2.0
            ind += 1
    return result

# Initial conditions
def initial_conditions(eq_type, d1, d2, unif2d_x, unif2d_y, rdist) :
    u0 = {}
    if ( eq_type == "linear") :
        neq = 1
        h0=0; h1=1; R=(b-a)/2/5; xc=(a+b)/2; yc=(c+d)/2;
# The original code had a periodic sine*sine distribution:
#        mounds = 4
#        u0_fun=lambda x,y:  h0 + h1*np.sin(mounds*x)*np.sin(mounds*y)  # Try the origin
# This one is a blob centered on the domain.
        u0_fun=lambda x,y:  h0+h1/2*(1+np.cos(np.pi*np.sqrt(np.square(x-xc)+np.square(y-yc))/R))*(np.sqrt(np.square(x-xc)+np.square(y-yc))<R)
        # u0_fun=lambda x,y: np.ones(unif2d_x[1].shape)
        for j in range(d2):
            for i in range(d1):
                local_pos_x = x_c[i] + 0.5*hx*unif2d_x[rdist[i,j]]
                local_pos_y = y_c[j] + 0.5*hy*unif2d_y[rdist[i,j]]
                u0[i,j,0] = u0_fun(local_pos_x,local_pos_y)

        print("len u0[0,0,0]", len(u0[i,j,0]))
    
    elif ( eq_type == "swe") :
        neq = 3
        h0=1000; h1=5; L=1e7; sigma=L/20;
        h0_fun  = lambda x,y : h0+h1*np.exp(-(np.square(x-L/2)+np.square(y-L/2))/(2*sigma**2))
        v0x_fun = lambda x,y : np.zeros(len(x))
        v0y_fun = lambda x,y : np.zeros(len(x))

        # set initial condition in the uniformly spaced quadrature points
        for j in range(d2):
            for i in range(d1):
                local_pos_x = x_c[i] + 0.5*hx*unif2d_x[rdist[i,j]]
                local_pos_y = y_c[j] + 0.5*hy*unif2d_y[rdist[i,j]]
                u0[i,j,0] = h0_fun(local_pos_x,local_pos_y)
                u0[i,j,1] = h0_fun(local_pos_x,local_pos_y) * v0x_fun(local_pos_x,local_pos_y)
                u0[i,j,2] = h0_fun(local_pos_x,local_pos_y) * v0y_fun(local_pos_x,local_pos_y)

    elif ( eq_type == "adv_sphere") :
        neq = 1
        th_c=np.pi/2; lam_c=3/2*np.pi; h0=1000; 
        rr = lambda lam, th : radius*np.arccos(np.sin(th_c)*np.sin(th)+np.cos(th_c)*np.cos(th)*np.cos(lam-lam_c))
        u0_fun = lambda lam, th: h0/2*(1+np.cos(np.pi*rr(lam,th)/radius))*(rr(lam,th)<radius/3).astype(np.double)
        
        # set initial condition in the uniformly spaced quadrature points
        for j in range(d2):
            for i in range(d1):
                local_pos_x = x_c[i] + 0.5*hx*unif2d_x[rdist[i,j]]
                local_pos_y = y_c[j] + 0.5*hy*unif2d_y[rdist[i,j]]
                u0[i,j,0] = u0_fun(local_pos_x,local_pos_y)

    elif ( eq_type == "swe_sphere") :
        neq = 3
        print(eq_type, " initial conditions not yet implemented ")
    else :
        print(eq_type, " not recognized, exiting ")
        sys.exit(-1)

    return neq, u0

# MAIN PROGRAM

# Calculate the degrees of the individual elements (np array)
rdist = degree_distribution("unif",d1,d2,r_max);

half_cell_x = (b-a)/d1/2;
half_cell_y = (d-c)/d2/2;
x_c=np.linspace(a+half_cell_x,b-half_cell_x,d1); # Cell centers in X
y_c=np.linspace(c+half_cell_y,d-half_cell_y,d2); # Cell centers in Y

# To support the variable length of the uniform space, we use lists
#
unif2d_x = {}
unif2d_y = {}

#
# The Kronecker product is used to form the tensor
for r in range(r_max+1):
     unif=np.linspace(-1,1,r+1)
     unif2d_x[r] = np.kron(unif,np.ones(r+1))
     unif2d_y[r] = np.kron(np.ones(r+1),unif)

#
# These are the uniform points used for visualization -- use number of quadrature points

unif=np.linspace(-1,1,n_qp_1D)
unif2d_visual_x = np.kron(unif,np.ones(n_qp_1D))
unif2d_visual_y = np.kron(np.ones(n_qp_1D),unif)

if quad_type == "leg":
# Gauss-Legendre quadrature
    [pts,wts]=np.polynomial.legendre.leggauss(n_qp_1D)
elif quad_type == "lob":
# Gauss-Lobatto quadrature
    scheme=qp.line_segment.gauss_lobatto(n_qp_1D)
    pts=scheme.points
    wts=scheme.weights
else:
    [pts,wts]=np.polynomial.legendre.leggauss(n_qp_1D)
    print (type,"unsupported quadrature rule, using Legendre")

#
# The Kronecker product is used to form the tensor of quadrature points
pts2d_x = np.kron(pts,np.ones(n_qp_1D))
pts2d_y = np.kron(np.ones(n_qp_1D),pts)
wts2d   = np.kron(wts,wts)

# Create the Vandermonde matrix for the modal to nodal conversion
V = {}
V_rect = {}
phi_val_cell={}
phi_grad_cell_x={}
phi_grad_cell_y={}
phi_val_bd_cell_n={}
phi_val_bd_cell_s={}
phi_val_bd_cell_e={}
phi_val_bd_cell_w={}

lower  = -np.ones(1)   # Corresponds to the negative boundary of a Legendre polynomial
upper  =  np.ones(1)   # Corresponds to the positive boundary of a Legendre polynomial
for r in range(r_max+1):
    num_coeff = r+1

# Determine the coefficients for the orthogonality
    coeffs = norm_coeffs(num_coeff)

# Square matrix for the modal-nodal transformations
    legvander2d = np.polynomial.legendre.legvander2d(unif2d_x[r],unif2d_y[r],[r,r])
    V[r] = np.multiply(legvander2d,coeffs)

# For the visualization we use a finer grid (# quadrature points) in the cell
# This means that the Vandermonde interpolation matrix is rectangular
    legvander2d = np.polynomial.legendre.legvander2d(unif2d_visual_x,unif2d_visual_y,[r, r])
    V_rect[r] = np.multiply(legvander2d,coeffs)

# Values and grads of basis functions in internal quadrature points, i.e.
# phi_val(i,j)=Phi_j(x_i) for i=1:dim_qp,j=1:dim. The x_i are the quadrature points,
    legvander2d = np.polynomial.legendre.legvander2d(pts2d_x,pts2d_y,[r, r])
    phi_val_cell[r] = np.multiply(legvander2d,coeffs)

# Gradients of the basis functions in internal quadrature points,
# they has two components due to the x and y derivatives
# Not a trivial calculation, since gradients of Legendre not supported in numpy
# and there is probably a much better way to calculate these...
    phi_grad_cell_x[r]=np.zeros((n_qp,num_coeff*num_coeff))
    phi_grad_cell_y[r]=np.zeros((n_qp,num_coeff*num_coeff))
    temp_vander_x = np.polynomial.legendre.legvander(pts2d_x,r)
    temp_vander_y = np.polynomial.legendre.legvander(pts2d_y,r)
    dLm_x         = np.zeros((n_qp,num_coeff))
    dLm_y         = np.zeros((n_qp,num_coeff))

    # Note the unclean way with which the coefficients are set, used, and then zeroed out again
    coeff = np.zeros(num_coeff)
    for m in range(num_coeff):
        coeff[m]=1.0
        dLm_x[:,m] = np.polynomial.legendre.legval(pts2d_x,np.polynomial.legendre.legder(coeff))
        dLm_y[:,m] = np.polynomial.legendre.legval(pts2d_y,np.polynomial.legendre.legder(coeff)) 
        coeff[m]=0.0

    for m in range(num_coeff):
        for n in range(num_coeff):
            phi_grad_cell_x[r][:,m*num_coeff+n]=np.multiply(temp_vander_y[:,n],dLm_x[:,m])
            phi_grad_cell_y[r][:,m*num_coeff+n]=np.multiply(temp_vander_x[:,m],dLm_y[:,n])

    phi_grad_cell_x[r] = np.multiply(phi_grad_cell_x[r],coeffs)
    phi_grad_cell_y[r] = np.multiply(phi_grad_cell_y[r],coeffs)

# Values of basis functions in boundary quadrature points, repeating for
# each face and degree;  dimensions: (num_quad_pts_per_face)x(cardinality)x(num_faces)
    phi_val_bd_cell_n[r] = np.multiply(np.polynomial.legendre.legvander2d(pts,   upper, [r,r]),coeffs)
    phi_val_bd_cell_s[r] = np.multiply(np.polynomial.legendre.legvander2d(pts,   lower, [r,r]),coeffs)
    phi_val_bd_cell_e[r] = np.multiply(np.polynomial.legendre.legvander2d(upper, pts,   [r,r]),coeffs)
    phi_val_bd_cell_w[r] = np.multiply(np.polynomial.legendre.legvander2d(lower, pts,   [r,r]),coeffs)

# Set initial conditions
neq, u0 = initial_conditions(eq_type, d1, d2, unif2d_x, unif2d_y, rdist)

# Check the modal-nodal-modal transformations
# this check only checks whether V has full rank, since m2n and n2m are inverse operations.

# u0_check=modal2nodal(d1,d2,neq,nodal2modal(d1,d2,neq,u0,V,rdist),V,rdist);

#for n in range(neq):
#    for j in range(d2):
#        for i in range(d1):
#            error = np.linalg.norm( u0_check[i,j,n] - u0[i,j,n] )
#            if ( error > 1.E-10 ) :
#               print( i, j, k, error, " too big")

# Create the local mass matrices and their inverses 
mass, inv_mass = compute_mass(phi_val_cell, wts2d, d1, d2, rdist, hx, hy, y_c, pts2d_y, eq_type);

# Now create U in modal space, where the time integration will be performed
u = nodal2modal(d1,d2,neq,u0,V,rdist)

for iter in range(N_it) :
    print("Performing iteration at time ", t)
    if (t+dt > T) :
        dt=T-t    # rewind for final iteration  (not sure why)
    t=t+dt

    if ( RK == 1 ) :
        rhs_u = compute_rhs(d1, d2, neq, u, rdist, n_qp_1D, phi_val_cell, phi_grad_cell_x, phi_grad_cell_y,
                            phi_val_bd_cell_n, phi_val_bd_cell_s, phi_val_bd_cell_e, phi_val_bd_cell_w, inv_mass,
                            hx, hy, wts, wts2d, radius, pts, pts, pts2d_x, pts2d_y, x_c, y_c, coriolis_fun, eq_type)
        for n in range(neq):
            for j in range(d2):
                for i in range(d1):
                    u[i,j,n] = u[i,j,n] + dt*rhs_u[i,j,n]

    elif ( RK == 2 ) :
        k1 = compute_rhs(d1, d2, neq, u, rdist, n_qp_1D, phi_val_cell, phi_grad_cell_x, phi_grad_cell_y,
                         phi_val_bd_cell_n, phi_val_bd_cell_s, phi_val_bd_cell_e, phi_val_bd_cell_w, inv_mass,
                         hx, hy, wts, wts2d, radius, pts, pts, pts2d_x, pts2d_y, x_c, y_c, coriolis_fun, eq_type)
#  Use a trick to avoid the need for another temporary variable for the U input
        for n in range(neq):
            for j in range(d2):
                for i in range(d1):
                    u[i,j,n] = u[i,j,n] + dt*k1[i,j,n]     # Add in dt/2 too much of k1

        k2 = compute_rhs(d1, d2, neq, u, rdist, n_qp_1D, phi_val_cell, phi_grad_cell_x, phi_grad_cell_y,
                         phi_val_bd_cell_n, phi_val_bd_cell_s, phi_val_bd_cell_e, phi_val_bd_cell_w, inv_mass,
                         hx, hy, wts, wts2d, radius, pts, pts, pts2d_x, pts2d_y, x_c, y_c, coriolis_fun, eq_type)

        for n in range(neq):
            for j in range(d2):
                for i in range(d1):
                    u[i,j,n] = u[i,j,n] + dt*(-k1[i,j,n]/2 + k2[i,j,n]/2)     # Yields: u+dt*(k1/2 + k2/2)

    elif ( RK == 3 ) :
        k1 = compute_rhs(d1, d2, neq, u, rdist, n_qp_1D, phi_val_cell, phi_grad_cell_x, phi_grad_cell_y,
                         phi_val_bd_cell_n, phi_val_bd_cell_s, phi_val_bd_cell_e, phi_val_bd_cell_w, inv_mass,
                         hx, hy, wts, wts2d, radius, pts, pts, pts2d_x, pts2d_y, x_c, y_c, coriolis_fun, eq_type)

#  Use a trick to avoid the need for another temporary variable for the U input
        for n in range(neq):
            for j in range(d2):
                for i in range(d1):
                    u[i,j,n] = u[i,j,n] + dt*k1[i,j,n]     # Yields : u+dt*k1

        k2 = compute_rhs(d1, d2, neq, u, rdist, n_qp_1D, phi_val_cell, phi_grad_cell_x, phi_grad_cell_y,
                         phi_val_bd_cell_n, phi_val_bd_cell_s, phi_val_bd_cell_e, phi_val_bd_cell_w, inv_mass,
                         hx, hy, wts, wts2d, radius, pts, pts, pts2d_x, pts2d_y, x_c, y_c, coriolis_fun, eq_type)

        for n in range(neq):
            for j in range(d2):
                for i in range(d1):
                    u[i,j,n] = u[i,j,n] + dt*(-3/4*k1[i,j,n] + k2[i,j,n]/4)     # Yields: u+dt*(k1/4 + k2/4)

        k3 = compute_rhs(d1, d2, neq, u, rdist, n_qp_1D, phi_val_cell, phi_grad_cell_x, phi_grad_cell_y,
                         phi_val_bd_cell_n, phi_val_bd_cell_s, phi_val_bd_cell_e, phi_val_bd_cell_w, inv_mass,
                         hx, hy, wts, wts2d, radius, pts, pts, pts2d_x, pts2d_y, x_c, y_c, coriolis_fun, eq_type)

        for n in range(neq):
            for j in range(d2):
                for i in range(d1):
                    u[i,j,n] = u[i,j,n] + dt*(-k1[i,j,n]/12 - k2[i,j,n]/12 + 2/3*k3[i,j,n]) # Yields: u+dt*(k1/6+k2/6+2*k1/3)

    elif ( RK == 4 ) :
        k1 = compute_rhs(d1, d2, neq, u, rdist, n_qp_1D, phi_val_cell, phi_grad_cell_x, phi_grad_cell_y,
                         phi_val_bd_cell_n, phi_val_bd_cell_s, phi_val_bd_cell_e, phi_val_bd_cell_w, inv_mass,
                         hx, hy, wts, wts2d, radius, pts, pts, pts2d_x, pts2d_y, x_c, y_c, coriolis_fun, eq_type)

#  Use a trick to avoid the need for another temporary variable for the U input
        for n in range(neq):
            for j in range(d2):
                for i in range(d1):
                    u[i,j,n] = u[i,j,n] + dt*k1[i,j,n]/2     # Yields : u+dt*k1/2

        k2 = compute_rhs(d1, d2, neq, u, rdist, n_qp_1D, phi_val_cell, phi_grad_cell_x, phi_grad_cell_y,
                         phi_val_bd_cell_n, phi_val_bd_cell_s, phi_val_bd_cell_e, phi_val_bd_cell_w, inv_mass,
                         hx, hy, wts, wts2d, radius, pts, pts, pts2d_x, pts2d_y, x_c, y_c, coriolis_fun, eq_type)

        for n in range(neq):
            for j in range(d2):
                for i in range(d1):
                    u[i,j,n] = u[i,j,n] + dt*(-k1[i,j,n]/2 + k2[i,j,n]/4)     # Yields: u+dt*k2/2

        k3 = compute_rhs(d1, d2, neq, u, rdist, n_qp_1D, phi_val_cell, phi_grad_cell_x, phi_grad_cell_y,
                         phi_val_bd_cell_n, phi_val_bd_cell_s, phi_val_bd_cell_e, phi_val_bd_cell_w, inv_mass,
                         hx, hy, wts, wts2d, radius, pts, pts, pts2d_x, pts2d_y, x_c, y_c, coriolis_fun, eq_type)

        for n in range(neq):
            for j in range(d2):
                for i in range(d1):
                    u[i,j,n] = u[i,j,n] + dt*(-k2[i,j,n]/2 + k3[i,j,n]) # Yields: u+dt*k3

        k4 = compute_rhs(d1, d2, neq, u, rdist, n_qp_1D, phi_val_cell, phi_grad_cell_x, phi_grad_cell_y,
                         phi_val_bd_cell_n, phi_val_bd_cell_s, phi_val_bd_cell_e, phi_val_bd_cell_w, inv_mass,
                         hx, hy, wts, wts2d, radius, pts, pts, pts2d_x, pts2d_y, x_c, y_c, coriolis_fun, eq_type)

        for n in range(neq):
            for j in range(d2):
                for i in range(d1):
                    # u=u+dt*1/6*k1+dt*1/3*k2+dt*1/3*k3+dt*1/6*k4
                    u[i,j,n] = u[i,j,n] + dt*(k1[i,j,n]/6 + k2[i,j,n]/3 - 2/3*k3[i,j,n] + 1/6*k4[i,j,n])

    else :
        print( RK, " is not a valid value (RK in {1,2,3,4}" )

    print(f'Iteration{iter} done')
    if ( iter%plot_freq == 0 or iter == N_it-1 ) :
        
        to_plot = modal2nodal(d1,d2,neq,u,V_rect,rdist)
        plot_solution(to_plot, x_c, y_c, n_qp_1D, d1, d2, neq, hx, hy)

# End of program
