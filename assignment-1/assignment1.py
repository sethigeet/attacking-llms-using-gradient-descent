import sys
from rich import print
from rich.table import Table
import numpy as np

with open(sys.argv[1]) as f:
    input_text = f.read()
    input_text = input_text.split("\n")

    c = list(map(float, input_text[0].split(" ")[1].split(",")))
    c = np.array(c)

    A = []
    b = []
    for constraint in input_text[1:]:
        if constraint == "":
            continue

        constraint = constraint.split("<=")
        coeffs, val = constraint[0], constraint[1]
        A.append(list(map(float, coeffs.split(","))))
        b.append(float(val))
    A = np.array(A)
    b = np.array(b)


def get_basic_feasible_solution(
    A: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    basic_variable_ids = np.zeros(len(A))
    for i in range(len(A)):
        basic_variable_ids[i] = len(A[0]) + i

    # make a space for each of the original variables, the newly introduced
    # artificial variables as well as `b`
    c = np.zeros(len(A[0]) + len(A) + 1)

    # new objective function: x4 + x5 + ...
    for i in range(len(A[0]), len(A[0]) + len(A)):
        c[i] = 1

    # make the initial tableau with the artificial variables
    tableau = np.array(
        [
            np.append(
                row,
                np.append(
                    np.array(([0] * i) + [1] + ([0] * (len(A) - i - 1))),
                    b[i],
                ),
            )
            for i, row in enumerate(A)
        ]
    )

    # compute initial the cost
    r = c.copy()
    for row in tableau:
        for i in range(len(row)):
            r[i] -= row[i]

    while not all([idx < len(A[0]) for idx in basic_variable_ids]):
        # find the pivot element
        pivot_col_idx = np.argmin(r[:-1])
        pivot_row_idx = np.argmax(tableau[:, pivot_col_idx] / tableau[:, -1])
        pivot_element = tableau[pivot_row_idx][pivot_col_idx]

        # set the new basis variable
        basic_variable_ids[pivot_row_idx] = pivot_col_idx

        # update the tableau according to the pivot
        tableau[pivot_row_idx] /= pivot_element
        for i in range(len(tableau)):
            if i == pivot_row_idx:
                continue
            tableau[i] -= tableau[pivot_row_idx] * tableau[i][pivot_col_idx]
        r -= tableau[pivot_row_idx] * r[pivot_col_idx]

    return tableau, np.array(basic_variable_ids, dtype="int")


def get_optimal_solution(
    tableau: np.ndarray,
    c: np.ndarray,
    basic_variable_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basic_variable_ids = basic_variable_ids.copy()
    tableau = tableau.copy()
    r = np.append(c.copy(), [0])

    # calculate the initial cost function
    for basic_variable_id in basic_variable_ids:
        row_idx = np.where(tableau[:, basic_variable_id] == 1)[0][0]
        factor = c[basic_variable_id] / tableau[row_idx][basic_variable_id]
        r -= tableau[row_idx] * factor

    while not np.all(r[:-1] >= 0):
        # find the pivot element
        pivot_col_idx = np.argmin(r[:-1])
        pivot_row_idx = np.argmax(tableau[:, pivot_col_idx] / tableau[:, -1])
        pivot_element = tableau[pivot_row_idx][pivot_col_idx]

        # set the new basis variable
        basic_variable_ids[pivot_row_idx] = pivot_col_idx

        # update the tableau according to the pivot
        tableau[pivot_row_idx] /= pivot_element
        for i in range(len(tableau)):
            if i == pivot_row_idx:
                continue
            tableau[i] -= tableau[pivot_row_idx] * tableau[i][pivot_col_idx]
        r -= tableau[pivot_row_idx] * r[pivot_col_idx]

    return tableau, r, basic_variable_ids


tableau, basic_variable_ids = get_basic_feasible_solution(A, b)

# remove the newly introduced artificial variables from the tableau
tableau = np.array([np.append(row[0 : len(A[0])], row[-1]) for row in tableau])

# find the optimal solution
tableau, c, basic_variable_ids = get_optimal_solution(tableau, c, basic_variable_ids)

# Print the required things
table = Table(title="Final Tableau")
for i in range(len(A[0])):
    table.add_column(f"x{i+1}", justify="center", style="green")
table.add_column("b", justify="center", style="magenta")

for i, row in enumerate(tableau):
    table.add_row(
        *list([str(round(item, 2)) for item in row]),
        end_section=(i + 1 == len(tableau)),
    )
table.add_row(*list([str(round(item, 2)) for item in c]), end_section=True)

print()
print(table)
print()
for i in range(len(A[0])):
    variable_ids = np.where(basic_variable_ids == i)[0]
    value = 0
    if len(variable_ids) != 0:
        variable_id = variable_ids[0]
        value = round(tableau[variable_id][-1], 2)

    print(f"[green]x{i+1}[/green] = [blue]{value}[/blue]")
print()
print(
    "[dim][red]NOTE:[/red] All the values have been rounded down to 2 decimal places!"
)
