'''
The code for my PHY407 final project: an implementation of the MISER algorithm to solve 
multidimensional integrals. If you're looking for the main running block of code, scroll
to the bottom and see the __main__ statement. The function of interest here is the one
named miser(). Everything else is fairly simple, and within the scope of PHY407. 

Author: Mitchell Barrett
'''

import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='serif')

def variance(N, f_vals):
    # a simple function to calculate the variance of a function over N samples
    return (sum(f_vals**2) / N) - (sum(f_vals) / N)**2

def circular_log(pts):
    # this function represents log(r) for r <= 1
    # it accomplishes this by multiplying by the unit_circle() function
    # this ensure calculated points fall within the circle of r = 1
    x_vals = pts[:,0]
    y_vals = pts[:,1]
    f_vals = np.log(np.sqrt(x_vals**2 + y_vals**2)) * unit_circle(pts)
    return f_vals

def unit_circle(pts):
    # this function defines a unit circle in R^2
    # it can take an array of arbitrary length, width 2 as its argument
    # this function is defined in a weird way, in order to make it possible to pass large arrays as arguments
    # returns an array of the values of the function for each point given as input
    x_vals = pts[:,0]
    y_vals = pts[:,1]
    f_vals = np.zeros_like(x_vals)
    for j in np.where(x_vals**2 + y_vals**2 <= 1):
        f_vals[j] = 1
    return f_vals

def unit_sphere(pts):
    # the same as unit_circle(), except for a sphere in R^3
    x_vals = pts[:,0]
    y_vals = pts[:,1]
    z_vals = pts[:,2]
    f_vals = np.zeros_like(x_vals)
    for j in np.where(x_vals**2 + y_vals**2 + z_vals**2 <= 1):
        f_vals[j] = 1
    return f_vals

def unit_nball(pts, ndim):
    # this is a generalization of the previous two functions
    # given ndim, this function defines a closed ball in R^ndim
    total = np.zeros_like(pts[:,0])
    f_vals = np.zeros_like(pts[:,0])
    for j in range(ndim):
        total += pts[:,j]**2
    for k in np.where(total <= 1):
        f_vals[k] = 1
    return f_vals

def random_points(corner1, corner2, npoints = 1):
    # generates a random point within a volume of arbitrary dimension
    # corner1 and corner2 should be numpy arrays, and are points on opposite corners of the volume
    # they can be lists if you want; lists just get converted to arrays
    # the volume will be assumed to be a square in 2D, a cube in 3D, a hypercube in 4D, etc.
   
    # ndim is the dimension of the space in which the point exists
    ndim = len(corner1)
    
    # pts will be an array of points in the [0,1]^ndim hypercube
    # it will have shape (number_of_points, number_of_dimensions)
    # in this way, pts[j] will be the j'th random point
    pts = np.random.rand(npoints, ndim)
    
    # pts will then be transformed so the points fit into the correct volume
    try: 
    # start by assuming the corners are numpy arrays, do easy math
        pts *= corner2 - corner1
        pts += corner1
    except TypeError: 
    # if the corners are passed as lists instead of arrays, a TypeError is raised and this block executes
        corner1 = np.array(corner1)
        corner2 = np.array(corner2)
        pts = random_points(corner1, corner2, npoints)
        
    return pts

def miser(N, lower, upper, func):
    # an algorithm for doing Monte Carlo integration with recursive stratified sampling
    # inputs:
        # N: int, number of points to be used for MC integration within the current volume
        # lower: array-like, lower corner of a rectangular volume
        # upper: array-like, upper corner of a rectangular volume
        # func: function, the function to be integrated
    # note that func will have to behave nicely with the output from random_points()
    
    # RETURNS
    # avg:          average value of the integral on the given volume
    # std:          standard deviation or error
    #               in implementation, this value is not important until the very end
    #               its returned value will be a good estimate of the error associated with the whole calculation
    # pts:          an array of all points used in the calculation
    # var_tilde:    variance / N 
    #               updated after each recursive step but not important at the end of the calculation
    
    # this function is given a number of points N and a volume of arbitrary dimension
    # some cutoff number of points is specified below
    # if N is below the cutoff, an MC integration is done
    # otherwise, the volume is recursively partitioned using the MISER algorithm
    # this will cluster the random points around the regions of highest variance
    
    # NOTE: this function will do 1D integrations, but you must pass the endpoints as lists of length 1
    

    
    # if lower and upper are passed as lists, we convert them to arrays
    # if they're passed as arrays, the following two lines do no harm and don't add any significant runtime
    lower = np.array(lower, dtype=np.float64)
    upper = np.array(upper, dtype=np.float64)
    
    # V is the total volume of this region
    V = np.prod(upper - lower)
    
    # cutoff is the number of points below which a simple Monte Carlo integration will be done
    # if N is greater than cutoff, the recursion will continue
    # one may change cutoff as one sees fit
    cutoff = 40
    if N <= cutoff:
        # the following lines do a standard Monte Carlo integration
        # pts are the N random points
        # fvals are the values of the function evaluated at every point in pts
        # avg, var_tilde, and std are the usual values calculated in the usual way
        pts = random_points(lower, upper, N)
        f_vals = func(pts)
        avg = (V / N) * sum(f_vals)
        var_tilde = variance(N, f_vals) / N
        std = V * np.sqrt(var_tilde)
        
    # if the number of points in our region is greater than cutoff, we want partition our volume
    else:
        M = 500
        # M is an intermediate number of Monte Carlo points
        # this number is used to make quick variance estimates over different volumes obtained by different partitions
        # it should not be set too high
    
        variances = []
        variances_V1 = []
        variances_V2 = []
        # the following loop calculates the summed variances of different partitions
        for j in range(len(lower)):
            # j represents the j'th coordinate axis
            test_new_lower = np.copy(lower); test_new_lower[j] += (upper[j] - lower[j]) / 2
            test_new_upper = np.copy(upper); test_new_upper[j] -= (upper[j] - lower[j]) / 2
            
            # V1 represents the lower volume, formed with the original lower corner and the new upper corner
            # V2 represents the lower volume, formed with the original upper corner and the new lower corner
            intermediate_pts_V1 = random_points(lower, test_new_upper, M)
            intermediate_f_vals_V1 = func(intermediate_pts_V1)
            intermediate_var_V1 = variance(M, intermediate_f_vals_V1)
            
            intermediate_pts_V2 = random_points(test_new_lower, upper, M)
            intermediate_f_vals_V2 = func(intermediate_pts_V2)
            intermediate_var_V2 = variance(M, intermediate_f_vals_V2)
            
            intermediate_variance = intermediate_var_V1 + intermediate_var_V2
            
            variances.append(intermediate_variance)
            variances_V1.append(intermediate_var_V1)
            variances_V2.append(intermediate_var_V2)
        
        # we choose to partition along the axis which minimizes the summed variances
        cut_index = np.where(np.array(variances) == min(variances))[0][0]
        new_lower = np.copy(lower); new_lower[cut_index] += (upper[cut_index] - lower[cut_index]) / 2
        new_upper = np.copy(upper); new_upper[cut_index] -= (upper[cut_index] - lower[cut_index]) / 2
        
        # define the new corners of the new volumes
        lower_V1 = lower
        upper_V1 = new_upper
        lower_V2 = new_lower
        upper_V2 = upper
        
        # identify the variances associated with the new volumes
        var_V1 = variances_V1[cut_index]
        var_V2 = variances_V2[cut_index]
        
        sigma_V1 = np.sqrt(var_V1)
        sigma_V2 = np.sqrt(var_V2)
        
        # we set the new N proportional to sigma1 / (sigma1 + sigma2)
        # if either sigma is 0, we simply take N to be the cutoff
        if sigma_V1 == 0:
            N_V1 = cutoff
            N_V2 = N - N_V1
        elif sigma_V2 == 0:
            N_V2 = cutoff
            N_V1 = N - N_V2
        else:
            N_V1 = int(np.round( N * (sigma_V1 / (sigma_V1 + sigma_V2)) ))
            N_V2 = N - N_V1
            
        # I had a division by zero error on a couple runs, so this is just to avoid that
        # it doesn't change the results at all, it may cause N to differ from the given value by a few points
        if N_V1 == 0 or N_V2 == 0:
            N_V1 += 1
            N_V2 += 1
        
        # run the recursion
        avg_V1, std_V1, pts_V1, var_tilde_V1 = miser(N_V1, lower_V1, upper_V1, func)
        avg_V2, std_V2, pts_V2, var_tilde_V2 = miser(N_V2, lower_V2, upper_V2, func)
        
        # the average value of the function across both volumes is just the summed averages
        # the associated error is only to make the returns work out, var_tilde is the interesting quantitiy
        # we append the points used to an array so that they may be plotted afterwards if desired
        avg = avg_V1 + avg_V2
        var_tilde = (1/4) * (var_tilde_V1 + var_tilde_V2)
        std = V * np.sqrt(var_tilde)
        #std = np.sqrt(std_V1**2 + std_V2**2)
        pts = np.append(pts_V1, pts_V2, axis=0)
    
    return avg, std, pts, var_tilde

def monteCarloMeanValue(N, a, b, f):
    # this function is standard Monte Carlo mean value integration in 1D
    # I took this function directly from lab 10 (since I wrote the function myself, I assume that's ok)
    # the main function used for my project is more involved, this one is simply for comparative purposes
    
    # monte carlo mean value method to be used for standard integrals of one variable (e.g y = f(x))
    # N is number of random samples
    # a and b are the endpoints of the domain (a > b)
    # f is the function to be evaluated
    
    # returns the value of the integral `I` and the standard deviation `std`
    
    # generate N random values between 0 and 1
    # put these values into the correct interval (multiply by interval length, shift by a)
    # pass values into f
    x_vals = np.random.rand(N) * (b - a) + a
    f_vals = f(x_vals)
    
    # calculate the integral with the given formula
    I = ((b - a) / N) * sum(f_vals)
    
    var = (1/N) * sum(f_vals**2) - ( (1/N) * sum(f_vals) )**2
    
    std = (b - a) * np.sqrt(var / N)
    
    return I, std

# run the code
if __name__ == '__main__':
    
    cl = [-1, -1]; cu = [1, 1] # "circle lower" and "circle upper"
    circle_avg, circle_std, circle_pts, circle_var = miser(25000, cl, cu, unit_circle)
    # we use 20000 points so that the resulting plot is readable
    # 20000 seems to be the sweet spot between too many and too few points
    # (plotted on a 6in x 6in plot with markersize = 1)
    
    plt.figure(figsize = (6,6))
    plt.plot(circle_pts[:,0], circle_pts[:,1], ls='', marker='o', markersize='1', color='k')
    plt.title('MISER algorithm on the 2D unit circle')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.savefig('miser_circle.png', dpi=150)
    
    sl = [-1, -1, -1]; su = [1, 1, 1] # "sphere lower" and "sphere upper"
    sphere_avg, sphere_std, sphere_pts, sphere_var = miser(100000, sl, su, unit_sphere)
    
    hs4l = [-1, -1, -1, -1]; hs4u = [1, 1, 1, 1] # "hypersphere 4d lower" and "hypersphere 4d upper"
    hs4_avg, hs4_std, hs4_pts, hs4_var = miser(100000, hs4l, hs4u, lambda pts: unit_nball(pts, ndim = len(hs4l)))
    
    log2d_avg, log2d_std, log2d_pts, log2d_var = miser(100000, cl, cu, circular_log)
    
    log_avg_miser, log_std_miser, log_pts_miser, log_var = miser(1000, [0], [1], np.log)
    log_avg_MCMV, log_std_MCMV = monteCarloMeanValue(1000, 0, 1, np.log)
    
    print(f'The area of the unit circle is {circle_avg} plus or minus {circle_std}.')
    print(f'The volume of the unit sphere is {sphere_avg} plus or minus {sphere_std}.')
    print(f'The volume of the 4D unit hypersphere is {hs4_avg} plus or minus {hs4_std}.')
    
    print('\n')
    
    print('Comparing MISER to mean value:')
    print(f'MV: the integral of the natural logarithm between 0 and 1 is {log_avg_MCMV} plus or minus {log_std_MCMV}')
    print(f'MISER: the integral of the natural logarithm between 0 and 1 is {log_avg_miser[0]} plus or minus {log_std_miser}')
    
    print('\n')
    
    print(f'The value of ln(r) evaluated in polar coordinates between r=0 and r=1 is {log2d_avg} plus or minus {log2d_std}')