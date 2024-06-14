"""
Functions for compatibility with MyGrad to allow automatic differentiation. Includes:
    get_matrix_function() which returns a mygrad compatible version of the _matrix_function() in myokit markov models.
    linalg_solve() which is a mygrad compatible matrix Ax=b equation solver.
"""
import myokit
import myokit.lib.markov
import numpy as np
import mygrad as mg


# =============================================================================
#
# =============================================================================


#  Code taken from https://github.com/myokit/myokit/blob/main/myokit/lib/markov.py on 2024/02/07 (last edit was commit https://github.com/myokit/myokit/commit/31fe43cea4af9efcfb0d428e056e5d82173170c3) under the following license. It has been changed to work with MyGrad rather than numpy and with comments to ignore some code for coverage calculations.
# =============================================================================
# BSD 3-Clause License
#
# Copyright (c) 2011-2017 Maastricht University. All rights reserved.
# Copyright (c) 2017-2020 University of Oxford. All rights reserved.
# (University of Oxford means the Chancellor, Masters and Scholars of the University of #Oxford, having an administrative office at Wellington Square, Oxford OX1 2JD, UK).
# Copyright (c) 2020-2024 University of Nottingham. All rights reserved.
# =============================================================================

# noinspection PyProtectedMember,PyListCreation,PyUnboundLocalVariable,PyUnusedLocal,GrazieInspection
def get_matrix_function(model):
    """
    Generates a MyGrad compatible _matrix_function for the automatic differentiation in the calculation of Markov model steady states. 
    Contains edits to swap out the numpy transition matrix with a MyGrad tensor.
    Removed any code that edits the model.
    Some edits to allow the code to work outside myokit.lib.markov
    Unnecessary code for the matrix function has been commented out.
    """
    # Create mapping from states to index
    state_indices = {}
    for k, state in enumerate(model._states):
        state_indices[state] = k

    # Get expressions for state & current variables, but with all
    # references to variables (except states and parameters) replaced by
    # inlined expressions.
    expressions = []
    for v in model._states:
        expressions.append(v.rhs().clone(expand=True, retain=model._inputs))
    if model._current is not None:
        current_expression = model._current.rhs().clone(
            expand=True, retain=model._inputs)

    # Create parametrisable matrices to evaluate the state & current
    # This checks that each state's RHS can be written as a linear
    # combination of the states, and gathers the corresponding multipliers.

    # Matrix of linear factors
    n = len(model._states)
    A = [[myokit.Number(0) for j in range(n)] for i in range(n)]

    # # List of transitions
    # T = set()

    # Populate A and T
    for row, e in enumerate(expressions):

        # Check if this expression is a linear combination of the states
        try:
            factors = myokit.lib.markov._linear_combination(e, model._states)
        except ValueError:  # pragma: no cover
            raise myokit.lib.markov.LinearModelError(
                'Unable to write expression as linear combination of'
                ' states: ' + str(e) + '.')

        # Scan factors
        for col, state in enumerate(model._states):
            factor = factors[col]
            if factor is not None:
                # Add factor to transition matrix
                A[row][col] = factor

                # # Store transition in transition list
                # if row != col:
                #     T.add((col, row))   # A is mirrored

    # Create a parametrisable matrix for the current
    B = [myokit.Number(0) for i in range(n)]
    if model._current is not None:
        try:
            factors = myokit.lib.markov._linear_combination(current_expression, model._states)
        except ValueError:  # pragma: no cover
            raise myokit.lib.markov.LinearModelError(
                'Unable to write expression as linear combination of'
                ' states: ' + str(e) + '.')

        for col, state in enumerate(model._states):
            factor = factors[col]
            if factor is not None:
                B[col] = factor

    # # Create list of transition rates and associated equations
    # T = list(T)
    # T.sort()
    # R = []
    # for i in range(len(A)):
    #     for j in range(len(A)):
    #         if (i, j) in T:
    #             R.append((i, j, A[j][i]))   # A is mirrored
    # del T

    #
    # Create function to create parametrisable matrices
    #
    model._model.reserve_unique_names('A', 'B', 'n', 'numpy')
    writer = myokit.numpy_writer()
    w = writer.ex
    head = 'def matrix_function('
    head += ','.join([w(p.lhs()) for p in model._inputs])
    head += '):'
    body = []
    body.append('A = mg.zeros((n, n), dtype=numpy.double)')
    zero = myokit.Number(0)
    for i, row in enumerate(A):
        for j, e in enumerate(row):
            if e != zero:
                body.append('A[' + str(i) + ',' + str(j) + '] = ' + w(e))
    body.append('B = mg.zeros(n, dtype=numpy.double)')
    for j, e in enumerate(B):
        if e != zero:
            body.append('B[' + str(j) + '] = ' + w(e))
    body.append('return A, B')
    code = head + '\n' + '\n'.join(['    ' + line for line in body])
    globl = {'numpy': np, 'n': n, 'mg': mg}
    local = {}

    exec(code, globl, local)
    # model._matrix_function = local['matrix_function']

    # #
    # # Create function to return list of transition rates
    # #
    # model._model.reserve_unique_names('R', 'n', 'numpy')
    # head = 'def rate_list_function('
    # head += ','.join([w(p.lhs()) for p in model._inputs])
    # head += '):'
    # body = []
    # body.append('R = []')
    # for i, j, e in R:
    #     body.append(
    #         'R.append((' + str(i) + ',' + str(j) + ',' + w(e) + '))')
    # body.append('return R')
    # code = head + '\n' + '\n'.join(['    ' + line for line in body])
    # globl = {'numpy': np}
    # local = {}
    # exec(code, globl, local)
    # model._rate_list_function = local['rate_list_function']

    return local['matrix_function']


def linalg_solve(A, B):
    """
    Solve the linear system Ax=B for x. Uses MyGrad tensors to ensure automatic differentiation works.
    Performs Gaussian elimination with partial pivoting followed by back substitution.

    Parameters
    ----------
    A : MyGrad tensor
        Square matrix A.
    B : MyGrad tensor
        A vector of length matching the size of A.

    Returns
    -------
    x : MyGrad tensor
        The solution to Ax=B with all derivative information included.
    """
    x = mg.zeros(B.shape, dtype=np.double)
    n = B.size
    pivot = list(range(n))
    for row in range(n - 1):
        # Partial pivoting
        for i in range(row + 1, n):
            if np.abs(A[pivot[row], row]) < np.abs(A[pivot[i], row]):
                pivot[row], pivot[i] = pivot[i], pivot[row]
        # Gaussian elimination
        for i in range(row + 1, n):
            ratio = A[pivot[i], row] / A[pivot[row], row]
            A[pivot[i], :] -= ratio * A[pivot[row], :]
            B[pivot[i]] -= ratio * B[pivot[row]]
    # Zero out any rounding errors in A
    for row in range(n):
        for i in range(row):
            A[pivot[row], i] = 0
    # Back substitution
    x[n - 1] = B[pivot[n - 1]] / A[pivot[n - 1], n - 1]

    for row in range(n - 2, -1, -1):
        summ = A[pivot[row], :] @ x
        x[row] = (B[pivot[row]] - summ) / A[pivot[row], row]
    return x
