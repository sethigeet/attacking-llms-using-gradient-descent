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

> [!NOTE]
> This script currently only supports minimization though adding support for maximization is trivial!

### More Efficient Method using Matrices

This logic has been coded up in `assignment1-v2.py`.

This method follows a similar approach to the first one but instead of carrying out row operations to solve the simplex tableau, we use matrices to improve the speed.

#### Procedure

1. Start by making the initial simplex tableau.
1. Split the tableau into `B` and `N` containing basic and non-basic variables respectively
1. Calculate the cost
1. Check if the problem is already optimized. If it is, end the program. Otherwise, continue.
1. Find the pivot element and hence find the variable which will be entering the basis and the one which will be exiting.
1. Swap the variables and update the tableau
1. Go back up to step 2.

> [!NOTE]
> This script currently only supports maximization though adding support for maximization is trivial!

> [!NOTE]
> The logic for this method was largely inspired by these [notes](https://personal.math.ubc.ca/~loew/m340/rsm-notes.pdf).

## Input Format

The input is mainly divided into two parts which are elaborated on below:

### Part 1

The first part defines the function to optimize. It follows the structure described below.

- It starts with the task i.e. whether to "maximize" or to "minimize".
- This is then followed by a space.
- This is followed by comma (`,`) separated numbers which are coefficients of each of the variables.

### Part 2

The second part consists of all the constraints under which the function is to be optimized in. Each of the constraints must be present on a new line. Each of them follow the structure described below.

- It starts with comma (`,`) separated numbers which are coefficients of each of the variables on the LHS of the constraint.
- This is followed by `<=`. (NOTE: Currently, the scripts only support this type of constraint).
- Finally, the constraint ends with the RHS of the constraint.

> [!CAUTION]
> The "space" characters in the input must be exactly as specified in the structures above. There must be no space present between the commas and the `<=` sign.

> [!IMPORTANT]
> Wherever specified, the coefficients of each of the variables must be specified. If you do not want the variable in the function, put a 0 there.

> [!TIP]
> Some example files can be found in the `examples/` folder!
