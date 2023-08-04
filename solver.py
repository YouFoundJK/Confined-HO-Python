import numpy as np

def shifted_inv_power(A, x, t = 0.):
    '''
    Shifter inverse power Eigen equation solver
    '''
    def solve(A, b):
        '''
        Solves linear system Ax = b using Gaussian Elimination
        '''
        A = A.astype(float).copy()
        b = b.astype(float).copy()
        m = len(A)

        # Perform forward elimination
        for k in range(m-1):
            # Find the row with the largest pivot
            max_row = max(range(k, m), key=lambda i: abs(A[i, k]))
            
            # Swap the rows if needed
            if max_row != k:
                A[[k, max_row]] = A[[max_row, k]]
                b[[k, max_row]] = b[[max_row, k]]
            
            # Check for singular matrix
            if A[k, k] == 0:
                raise ValueError("Singular matrix")
            
            # Eliminate the lower entries
            for i in range(k+1, m):
                factor = A[i, k] / A[k, k]
                A[i] -= factor * A[k]
                b[i] -= factor * b[k]

        # Perform backward substitution
        x = [0] * m
        for i in range(m-1, -1, -1):
            x[i] = (b[i] - sum(A[i, j] * x[j] for j in range(i+1, m))) / A[i, i]

        return np.array(x)


    tol = 1e-8
    max_iter = 100
    m, n = A.shape
    iter = 0
    err = np.inf
    
    I = np.eye(m)       # precomputing I to reuse it for optimisation
    while err > tol and iter < max_iter:
        y = solve(A - t*I, x)

        v = y / np.linalg.norm(y)
        lam = v.dot(A.dot(v))
        err = np.linalg.norm((A - lam*np.eye(m)).dot(v))
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
