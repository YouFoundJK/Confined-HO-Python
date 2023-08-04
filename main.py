import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
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




L = 6               # half of domain size
N = 500 

# Plot initializing
plt.figure(figsize=(12,10))
x = np.linspace(-1.8, +1.8, round(N/2))
plt.plot(x, Vpot(x), lw=2, ls="-.", label="Harmonic Potential", color='black')


x = np.linspace(-L ,L, N)
h = x[1]-x[0]

# Hamiltonian Matrix
T = np.array([[0, -2, 1] if i == 0 else [1, -2, 0] if i == N-3 else [1, -2, 1] for i in range(N-2)]) 
V = np.array([Vpot(x[i+1]) for i in range(N-2)]) 
H = tridiag_diag_add(-T/(h**2), V)


# Eigen values using shifted Inverse power 
z, vec, iterations, error = shifted_inv_power(H, np.linspace(0.1,1,N-2), 2.5)

# Normalizing the eigenvector
norm = np.sqrt(trapezoidal_rule(vec **2, x[1:-1]))
vec = vec / norm

print("Eigenvalue:", z)
print("Iterations:", iterations)
print("Error:", error)


# Plotting the graphs
y = np.concatenate(([0], vec, [0]))         # matching the np.array shapes
plt.plot(x, y, lw=3, label="0", color="blue")             # Use label directly instead of formatting

plt.plot(x, psi_hermite(x, 1), lw=2, label="Hermite", ls='--', color="orange")


plt.xlabel('x', size=14)
plt.ylabel('$\psi$(x)',size=14)
plt.legend()
plt.title('Confined & Simple HO solution',size=14)
plt.grid() # Add major grid lines
plt.minorticks_on() # Enable minor ticks
plt.grid(which='minor', alpha=0.2) # Add minor grid lines with lower opacity
plt.tight_layout() # Adjust the spacing of the plot
plt.show()

