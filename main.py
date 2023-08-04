import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from solver import shifted_inv_power, tridiag_diag_add

''' PART 1'''

def Vpot(x):
    ''' Renormalized Potential'''
    return x**2

def trapezoidal_rule(F, X):
    ''' 
    F - function to integrate as numpy array
    X - Uniform domain interval
    '''
    dX = X[1]-X[0]
    # compute the sum of the areas of the trapezoids using the formula
    # (h/2) * (f[0] + 2*f[1] + ... + 2*f[N-1] + f[N])
    s = dX * (F[0] + 2 * np.sum(F[1:-1]) + F[-1]) / 2

    # return the approximation
    return s

def psi_hermite(x, n): # Hermite polynomial solution
    coeff = 1 / np.sqrt(2**n * sp.factorial(n) * np.sqrt(np.pi))
    hermite = sp.hermite(n)
    norm = np.sqrt(trapezoidal_rule(coeff**2 * hermite(x)**2 * np.exp(-x**2), x))       # L2 norm
    print(norm)
    return coeff * hermite(x) * np.exp(-x**2 / 2) / norm                        # Normalized wavefunction



a = -6  # lower limit of the domain
b = +6  # upper limit of the domain
N = 500 

x = np.linspace(a,b,N)
h = x[1]-x[0]


# Hamiltonian Matrix
T = np.array([[0, -2, 1] if i == 0 else [1, -2, 0] if i == N-3 else [1, -2, 1] for i in range(N-2)]) 
V = np.array([Vpot(x[i+1]) for i in range(N-2)]) 
# print("Optimized \n",T)
H = tridiag_diag_add(-T/(h**2), V)
# print("Optimized \n",H)

'''   PART II '''

# Eigen values using shifted Inverse power 
z, vec, iterations, error = shifted_inv_power(H, np.linspace(0.1,1,N-2), 0.5)

# Normalizing the solution
norm = np.sqrt(trapezoidal_rule(vec **2, x[1:-1]))
vec = vec / norm

print("Eigenvalue:", z)
print("Iterations:", iterations)
print("Error:", error)

'''   PART III '''

# Plotting the graphs
plt.figure(figsize=(12,10))
for i in range(1):
# for i in range(len(z)):
    y = []
    # y = np.append(y,vec[:,z[i]])
    y = np.append(y,vec)
    y = np.append(y,0)
    y = np.insert(y,0,0)
    plt.plot(x,y,lw=3, label="{} ".format(i))
    plt.xlabel('x', size=14)
    plt.ylabel('$\psi$(x)',size=14)

plt.plot(x,psi_hermite(x,0),lw=4, label="Hermite", ls='--')

plt.legend()
plt.title('normalized wavefunctions for a harmonic oscillator using finite difference method',size=14)
plt.show()
