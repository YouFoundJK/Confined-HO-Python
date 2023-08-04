import numpy as np
import matplotlib.pyplot as plt
from solver import shifted_inv_power

''' PART 1'''

def Vpot(x):
    ''' Renormalized Potential'''
    return x**2


a = -6  # lower limit of the domain
b = +6  # upper limit of the domain
N = 7 

x = np.linspace(a,b,N)
h = x[1]-x[0]


# Hamiltonian Matrix
T = np.array([[-2 if i == j else 1 if np.abs(i-j) == 1 else 0 for j in range(N-2)] for i in range(N-2)])
V = np.array([[Vpot(x[i+1]) if i == j else 0 for j in range(N-2)] for i in range(N-2)])
H = -T/(h**2) + V
# print(H)


'''   PART II '''

# Eigen values using shifted Inverse power 
z, vec, iterations, error = shifted_inv_power(H, np.linspace(0.1,1,N-2), 2.5)

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
plt.legend()
plt.title('normalized wavefunctions for a harmonic oscillator using finite difference method',size=14)
# plt.show()
