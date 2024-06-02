# Assignment 1 - Simplex Method for Optimization

This assignment is about writing the logic for solving a linear programming problem of optimization (maximization or minimization) using the simplex method.

## Approach

### Basic Gauss-Jordan Method

This logic has been coded up in `assignment1.py`.

This method follows the usual method followed by hand to make and solve the simplex tableau.

#### Procedure

1. Firstly, find a basic feasible solution for the problem.
1. Now start by making the initial simplex tableau.
1. Calculate the cost
1. Check if the problem is already optimized. If it is, end the program. Otherwise, continue.
1. Find the pivot element and hence find the variable which will be entering the basis and the one which will be exiting.
1. Swap the variables and update the tableau
1. Go back up to step 3.

> ![NOTE]
> This script currently only supports minimization though adding support for maximization is trivial!
