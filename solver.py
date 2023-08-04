import numpy as np

def tridiag_matvec(A, x):

  d = A[:,1]        # vector of main diagonal elements of A
  s = A[0,2]        # scalar value of subdiagonal or superdiagonal elements of A
  n = len(d)        # size of matrix A
  y = [0] * n       # initialize result vector y
  
  # first element of y
  y[0] = d[0] * x[0] + s * x[1]
  
  # middle elements of y
  for i in range(1, n-1):
    y[i] = s * x[i-1] + d[i] * x[i] + s * x[i+1]
  
  # last element of y
  y[n-1] = s * x[n-2] + d[n-1] * x[n-1]
  
  return y


def tridiag_diag_add(Z, x):
    # Z is a numpy array of size m x m, representing a tridiagonal matrix
    # x is a diagonal matrix
    Z = Z.astype(float).copy()
    if not isinstance(x, np.ndarray):
        x=x*np.ones(len(Z), dtype = float)
    Z[:,1]+=x[:]
    return Z


def shifted_inv_power(A, x, t = 0.):
    '''
    Shifter inverse power Eigen equation solver
    '''
    def solve(A, b):
        '''
        Solves linear system Ax = b using Gaussian Elimination, leveraging the fact 
        that A is a tridiagonal matrix (Pivot and swapping avoided).
        A: tridiagonal matrix
        b: right-side vector
        '''
        A = A.astype(float).copy()
        b = b.astype(float).copy()
        m = len(A)


        # Perform forward elimination
        for k in range(m-1):
            # Check for singular matrix
            if A[k, 1] == 0:
                raise ValueError("Singular matrix")
            
            # Eliminate the lower entry
            factor = A[k+1, 0] / A[k, 1]
            A[k+1,:-1] -= factor * A[k,1:]
            b[k+1] -= factor * b[k]

        
        # Perform backward substitution
        x = [0] * m
        x[m-1] = b[m-1] / A[m-1, 1]
        for i in range(m-2, -1, -1):
            x[i] = (b[i] - A[i, 2] * x[i+1]) / A[i, 1]
        

        return np.array(x)


    tol = 1e-8
    max_iter = 100
    m, n = A.shape
    iter = 0
    err = np.inf
    
    

    # I = np.eye(m)       # precomputing I to reuse it for optimisation
    while err > tol and iter < max_iter:
        y = solve(tridiag_diag_add(A, -t), x) 

        v = y / np.linalg.norm(y)
        lam = v.dot(tridiag_matvec(A, v))
        err = np.linalg.norm((tridiag_matvec(tridiag_diag_add(A, -lam), v))) 
        x = v

        iter += 1

    return lam, v, iter, err


if __name__ == '__main__':
    # A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # x = np.array([0, 1, 1, 0])
    A = np.array([[2, 3], [0, 2]])
    x = np.array([0, 1])


    # Print the results

    e_value, e_vector, iterations, error = shifted_inv_power(A, x)
    print("Eigenvalue:", e_value)
    print("Eigenvector:", e_vector)
    print("Iterations:", iterations)
    print("Error:", error)
