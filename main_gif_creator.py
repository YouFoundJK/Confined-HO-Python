import numpy as np
from time import sleep
import scipy.special as sp
from os import remove as rm
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from solver import shifted_inv_power, tridiag_diag_add


def trapezoidal_rule(F, X):
    ''' 
    F - function to integrate as numpy array
    X - Uniform domain interval
    '''
    # (dx/2) * (f[0] + 2*f[1] + ... + 2*f[N-1] + f[N])
    return (X[1]-X[0]) * (F[0] + 2 * np.sum(F[1:-1]) + F[-1]) / 2

# Renormalised Harmonic Oscillator Potential
Vpot = lambda x: x**2           

# Hermite polynomials with precalculated L2 norm coeff
psi_hermite = lambda x, n: (1 / np.sqrt(2**n * sp.factorial(n) 
                                        * np.sqrt(np.pi))) * sp.hermite(n)(x) * np.exp(-x**2 / 2)



def worker(L):
    '''
    png creater for parallel processing. L is half the domain size.
    '''
    N = 500 

    plt.figure(figsize=(15, 10))
    col=['orange', 'blue', 'green']
    x = np.linspace(-7, 7, N)
    [plt.plot(x, psi_hermite(x, order), lw=2, label=f"Hermite order: {order}", ls='--', color=col[order]) for order in range(3)]
    x=x[round(N/2-70):round(N/2+70)] 
    print(x[0],x[-1])
    plt.plot(x, Vpot(x), lw=2, ls="-.", label="Harmonic Potential", color='black')

    for order in range(3):
        x = np.linspace(-L ,L, N)
        h = x[1]-x[0]


        # Hamiltonian Matrix
        T = np.array([[0, -2, 1] if i == 0 else [1, -2, 0] if i == N-3 else [1, -2, 1] for i in range(N-2)]) 
        V = np.array([Vpot(x[i+1]) for i in range(N-2)]) 
        H = tridiag_diag_add(-T/(h**2), V)


        # Eigen values using shifted Inverse power 
        z, vec, iterations, error = shifted_inv_power(H, np.linspace(0.1,1,N-2), 0.5 + 2 * order)

        # Normalizing the eigenvector
        norm = np.sqrt(trapezoidal_rule(vec **2, x[1:-1]))
        vec = vec / norm

        # print("Shift value: ",0.5 + 2 * order,
        #     "\nEigenvalue: ", z, " in iterations: ", iterations, "with Error: ", error)

        # Plotting the graphs
        
        y = np.concatenate(([0], vec, [0]))             # matching the np.array shapes
    
        plt.plot(x, y, lw=3, label=f"E ={z:.3f}", color=col[order])            # Use label directly instead of formatting

        plt.axvline(x=L, linestyle='--', color='black')          
        plt.axvline(x=-L, linestyle='--', color='black')          



    plt.xlabel('x', size=14)
    plt.ylabel('$\psi$(x)',size=14)
    plt.legend()
    plt.title('Confined & Simple HO solution',size=14)
    plt.grid()                          # Major grid lines
    plt.minorticks_on()                 # Minor ticks
    plt.grid(which='minor', alpha=0.2)  
    plt.tight_layout() 
    # plt.show()

    filename = f'./image_{L}.png'

    plt.savefig(filename)
    file_list.append(filename)
    plt.close()

    return filename


if __name__ == '__main__':

    worker(3)
    sleep(1000)
    file_list = []
    file_list = Parallel(n_jobs=10, verbose=True)(delayed(worker)(L) for L in np.linspace(1.0, 4.0, 200))
    sleep(5)
    temp = []
    temp = Parallel(n_jobs=10, verbose=True)(delayed(worker)(L) for L in np.linspace(4.1, 7, 50))
    file_list.extend(temp)


    with imageio.get_writer("Confined_HO.gif", mode="I", duration=120) as writer:
        for filename in file_list:
            image = imageio.imread(filename)
            writer.append_data(image)
        
            # remove the image files
            rm(filename)
    
    sleep(5)
