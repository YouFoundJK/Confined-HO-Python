import numpy as np

def tridiag_matvec(A, x):
    '''
    Multipling tridiagonal matrix A (defined only by its non zero terms) with a vector x.
    '''
    d = A[:,1]        # vector of main diagonal elements of A
    s = A[0,2]        # scalar value of subdiagonal or superdiagonal elements of A
    
    # create shifted versions of x
    x_shift_left = np.roll(x, -1)       # shift x one position to the left
    x_shift_right = np.roll(x, 1)       # shift x one position to the right
    
    # multiply each element of x by s and add them
    x_sub_super = s * (x_shift_left + x_shift_right)
    
    # multiply each element of d by the corresponding element of x and add them
    x_main = d * x
    return x_sub_super + x_main

def tridiag_diag_add(Z, x):
    '''
    Adding tridiagonal matrix Z (defined only by itstridiagonal terms) with a diagonal matrix
    (defined either as a scalar - when diag is same or as a vector of only diag non elements)
    '''
    Z = Z.astype(float).copy()
    if not isinstance(x, np.ndarray):
        x=x*np.ones(len(Z), dtype = float)
    Z[:,1]+=x[:]
    return Z


def shifted_inv_power(A, x, t = 0.):
    '''
    Shifted inverse power Eigen equation solver
    '''
    def solve(A, b):
        '''
        Solves linear system Ax = b using Gaussian Elimination, leveraging the fact 
        that A is a tridiagonal matrix (Pivot, swapping avoided).
        A: tridiagonal matrix
        b: right-side vector
        '''
        A = A.astype(float).copy()
        b = b.astype(float).copy()
        m = len(A)

        # Perform forward elimination
        for k in range(m-1):
            # Check for singular pivot
            if A[k, 1] == 0:
                raise ValueError("Singular pivot")
            
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


    tol = 1e-8          # set tolerance in error
    max_iter = 100      # set cap on iteration number

    iter = 0
    err = np.inf
    while err > tol and iter < max_iter:
        # numerical more stable than directly finding inv(A)*x
        y = solve(tridiag_diag_add(A, -t), x)   

        v = y / np.linalg.norm(y)
        lam = v.dot(tridiag_matvec(A, v))
        err = np.linalg.norm((tridiag_matvec(tridiag_diag_add(A, -lam), v))) 
        x = v

        iter += 1           # counter

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
