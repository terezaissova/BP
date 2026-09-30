import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power

# definice symbolů (reálné a komplexní)
k, L, Z0, S = sp.symbols('k L Z0 S', real=True)
P0, W0, PL, WL, C1, C2 = sp.symbols('P0 W0 PL WL C1 C2')
i = sp.I

# definice konkrétních hodnot
L_val = 1
SA_val = sp.pi
SB_val = 4 * sp.pi
S0_val = sp.pi
Sd_val = sp.pi # průřez poruchy
c0 = 343
rho0 = 1.21
Z0_val = rho0 * c0
N = 10

def substitute_matrix(matrix, length, area):
    """
    Vezme obecnou matici M a dosadí za symboly L a S konkrétní hodnoty.
    """
    return matrix.subs({L: length, S: area})

def matrix_one_cell(M):
    """
    Vytvoří matici přenosu pro jednu buňku.
    """
    M1 = substitute_matrix(M, L_val / 4, SA_val)
    M2 = substitute_matrix(M, L_val / 2, SB_val)
    M3 = substitute_matrix(M, L_val / 4, SA_val)
    M_cell = M1 * M2 * M3
    return M_cell

def matrix_defect(M):
    """
    Vytvoří matici přenosu pro poruchu, takže rovnou rouru.
    """
    M_def = substitute_matrix(M, L_val, Sd_val)
    return M_def

def matrix_complete(M, pos_def):
    """
    Vytvoří matici kompletní struktury N+1 buněk s jednou poruchou na místě pos_def.
    """
    M_one_cell = matrix_one_cell(M)
    M_def = matrix_defect(M)
    M_comp = matrix_power(M_one_cell, pos_def-1) @ M_def @ matrix_power(M_one_cell, N-pos_def+1)
    return M_comp

def get_transmission(M):
    """
    Z matice vrátí přenos T jako funkci k.
    """
    M_11 = M[0, 0]
    M_12 = M[0, 1]
    M_21 = M[1, 0]
    M_22 = M[1, 1]
    numerator = 2 * S0_val * Z0_val * (M_11 * M_22 - M_12 * M_21)
    denominator = S0_val**2 * M_12 - S0_val * Z0_val * M_11 - S0_val * Z0_val * M_22 + Z0_val**2 * M_21
    t_sym = numerator / denominator
    T = t_sym.subs({Z0: Z0_val})
    return sp.lambdify(k, T, modules='numpy')

def plot_transmission(t, min, max):
    k_vals = np.linspace(min, max, 1000)
    t_vals = t(k_vals)
    t_abs = np.abs(t_vals)
    plt.figure(figsize=(10, 5))
    plt.plot(k_vals, t_abs, color='blue', label='|T|')
    plt.title("Transmission")
    plt.xlabel("k")
    plt.ylabel("|T|", rotation=0)
    plt.xlim(min, max)
    plt.grid(True)
    plt.legend()
    plt.show()


# rovnice
eq1 = sp.Eq(P0, C1 + C2)
eq2 = sp.Eq(W0, (S / Z0) * (C1 - C2))
eq3 = sp.Eq(PL, C1 * sp.exp(i * k * L) + C2 * sp.exp(-i * k * L))
eq4 = sp.Eq(WL, (S / Z0) * (C1 * sp.exp(i * k * L) - C2 * sp.exp(-i * k * L)))

# řešení soustavy, prvky vypisovat např. s1[0][C1]
s1 = sp.solve([eq1, eq2, eq3, eq4], [PL, WL, C1, C2], dict=True)

# srovnání jednotlivých koeficientů a vybrání prvků do matice
PL = s1[0][PL]
PL_collected = sp.collect(sp.expand(PL), [P0, W0])
WL = s1[0][WL]
WL_collected = sp.collect(sp.expand(WL), [P0, W0])

PL_coeff_P0 = PL_collected.coeff(P0)
PL_coeff_W0 = PL_collected.coeff(W0)
WL_coeff_P0 = WL_collected.coeff(P0)
WL_coeff_W0 = WL_collected.coeff(W0)

M11 = sp.simplify(PL_coeff_P0.rewrite(sp.cos))
M12 = sp.simplify(PL_coeff_W0.rewrite(sp.sin))
M21 = sp.simplify(WL_coeff_P0.rewrite(sp.sin))
M22 = sp.simplify(WL_coeff_W0.rewrite(sp.cos))

# matice pro rovnou rouru obecně
M = sp.Matrix([[M11, M12],
               [M21, M22]])
print("Matice pro rovnou rouru obecně:")
sp.pprint(M)

# přenos pro rovnou rouru
# t = get_transmission(substitute_matrix(M, L_val, SA_val))
# plot_transmission(t, 0, 10)

# přenos pro jednu buňku jako funkce k
M_one_cell = matrix_one_cell(M)
t_one_cell = get_transmission(M_one_cell)
plot_transmission(t_one_cell, 0, 10)

# přenos pro N buněk jako funkce k
M_n_cells = matrix_power(M_one_cell, N)
t_n_cells = get_transmission(M_n_cells)
plot_transmission(t_n_cells, 0, 5)

# přenos pro N+1 buněk s jednou poruchou jako funkce k
pos_def = 5 # pozice poruchy
M_complete = matrix_complete(M, pos_def)
t_complete = get_transmission(M_complete)
plot_transmission(t_complete, 0, 5)