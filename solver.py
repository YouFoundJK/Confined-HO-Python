import numpy as np

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
            if A[k, k] == 0:
                raise ValueError("Singular matrix")
            
            # Eliminate the lower entry
            factor = A[k+1, k] / A[k, k]
            # print( "\n", factor, '=', A[k+1, k], '/', A[k, k])

            # print( "\n", A[k+1],'-', factor, '*', A[k])
            A[k+1] -= factor * A[k]
            b[k+1] -= factor * b[k]
        

        # Perform backward substitution
        x = [0] * m
        x[m-1] = b[m-1] / A[m-1, m-1]
        for i in range(m-2, -1, -1):
            x[i] = (b[i] - A[i, i+1] * x[i+1]) / A[i, i]

        # print( "\n", x)
        return np.array(x)


    tol = 1e-8
    max_iter = 3
    m, n = A.shape
    iter = 0
    err = np.inf
    
    

    I = np.eye(m)               # precomputing I to reuse it for optimisation
    while err > tol and iter < max_iter:
        y = solve(A - t*I, x)

        v = y / np.linalg.norm(y)
        lam = v.dot(A.dot(v))
        print("\n", A)
        err = np.linalg.norm((A - lam*I).dot(v))
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
