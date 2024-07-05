import numpy as np

from project.funcs.base import vcol, vrow

# We create the kernel functions. Since the kernel functions may need additional
# parameters, we create a function that creates on the fly the required kernel
# function.


def poly_kernel(degree, c):
    def poly_kernel_func(D1, D2):
        return (np.dot(D1.T, D2) + c) ** degree

    return poly_kernel_func


def rbf_kernel(gamma):
    def rbf_kernel_func(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that
        # |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z)

    return rbf_kernel_func
