import numpy as np
import matplotlib.pyplot as plt
from solver import shifted_inv_power

''' PART 1'''

def Vpot(x):
    ''' Renormalized Potential'''
    return x**2


a = -6  # lower limit of the domain
b = +6  # upper limit of the domain
N = 500 

x = np.linspace(a,b,N)
h = x[1]-x[0]


# Kinetic Energy Matrix using list comprehension 
# T = np.zeros((N-2)**2).reshape(N-2,N-2)
# for i in range(N-2):
#     for j in range(N-2):
#         if i==j:
#             T[i,j]= -2
#         elif np.abs(i-j)==1:
#             T[i,j]=1
#         else:
#             T[i,j]=0

T = np.array([[-2 if i == j else 1 if np.abs(i-j) == 1 else 0 for j in range(N-2)] for i in range(N-2)])


# Potential Energy Matrix
# V = np.zeros((N-2)**2).reshape(N-2,N-2)
# for i in range(N-2):
#     for j in range(N-2):
#         if i==j:
#             V[i,j]= Vpot(x[i+1])
#         else:
#             V[i,j]=0

V = np.array([[Vpot(x[i+1]) if i == j else 0 for j in range(N-2)] for i in range(N-2)])


# Hamiltonian Matrix
H = -T/(h**2) + V
# print(H.shape)
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
plt.show()