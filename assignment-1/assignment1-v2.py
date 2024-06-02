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


def get_optimal_solution(
    A: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    num_eqns = len(A)
    A = np.array(
        [
            np.append(row, [0] * i + [1] + [0] * (num_eqns - i - 1))
            for i, row in enumerate(A)
        ]
    )
    c = np.append(c, [0] * num_eqns)
    x = np.zeros(num_eqns * 2)
    for i, j in enumerate(range(num_eqns, num_eqns * 2)):
        x[j] = b[i]

    non_basic_variable_ids = np.array([i for i in range(num_eqns)])
    basic_variable_ids = np.array([i for i in range(num_eqns, num_eqns * 2)])

    while True:
        # calculate required matrices
        B = np.array([[row[i] for i in basic_variable_ids] for row in A])
        N = np.array([[row[i] for i in non_basic_variable_ids] for row in A])
        cB = np.array([c[i] for i in basic_variable_ids])
        cN = np.array([c[i] for i in non_basic_variable_ids])
        xB = np.dot(np.linalg.inv(B), b)

        # select variable will enter the basis
        # NOTE: the entering variable is decided using Blanc's rule
        y = np.dot(cB, np.linalg.inv(B))
        zN = np.dot(y, N) - cN
        entering_var_id = None
        for i, var_id in enumerate(non_basic_variable_ids):
            if zN[i] < 0:
                entering_var_id = var_id
                break

        # we are already at the optimal solution
        if entering_var_id is None:
            break

        d = np.dot(np.linalg.inv(B), A[:, entering_var_id])
        t = np.min(xB / d)
        exiting_var_id = basic_variable_ids[np.argmin(xB / d)]

        # update the variable values
        for i, var_id in enumerate(basic_variable_ids):
            x[var_id] = xB[i] - d[i] * t

        # swap the variables
        basic_variable_ids[np.where(basic_variable_ids == exiting_var_id)[0][0]] = (
            entering_var_id
        )
        non_basic_variable_ids[
            np.where(non_basic_variable_ids == entering_var_id)[0][0]
        ] = exiting_var_id

        temp = x[entering_var_id]
        x[entering_var_id] = x[exiting_var_id]
        x[exiting_var_id] = temp

        x[entering_var_id] = t

        basic_variable_ids = np.sort(basic_variable_ids)
        non_basic_variable_ids = np.sort(non_basic_variable_ids)

    final_tableau = []
    for i in range(num_eqns):
        if i in basic_variable_ids:
            final_tableau.append(B[:, np.where(basic_variable_ids == i)[0][0]])
        elif i in non_basic_variable_ids:
            final_tableau.append(N[:, np.where(non_basic_variable_ids == i)[0][0]])
    final_tableau.append(xB)
    final_tableau = np.transpose(final_tableau)
    return final_tableau, x


tableau, x = get_optimal_solution(A, b, c)
z = np.dot(x[range(len(A))], c)

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
print()

for i in range(len(A)):
    print(f"[green]x{i+1}[/green] = [blue]{round(x[i], 2)}[/blue]")
print(f"[green]z[/green] = [blue]{round(z, 2)}[/blue]")

print()
print("[dim][red]NOTE:[/red] All the values have been rounded down!")
