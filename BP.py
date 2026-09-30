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
rA_val = 1
rB_val = 2
r0_val = 1
c0 = 343
rho0 = 1.21
Z0_val = rho0 * c0
N = 10

def substitute_matrix(matrix, length, radius):
    """
    Vezme obecnou matici M a dosadí za symboly L a S konkrétní hodnoty.
    """
    return matrix.subs({L: length, S: sp.pi * radius**2})

def matrix_one_cell(M):
    """
    Vytvoří matici přenosu pro jednu buňku.
    """
    M1 = substitute_matrix(M, L_val / 4, rA_val)
    M2 = substitute_matrix(M, L_val / 2, rB_val)
    M3 = substitute_matrix(M, L_val / 4, rA_val)
    M_cell = M1 * M2 * M3
    return M_cell

def matrix_defect(M, L_def, r_def):
    """
    Vytvoří matici přenosu pro poruchu, takže rovnou rouru.
    """
    M_def = substitute_matrix(M, L_def, r_def)
    return M_def

def matrix_complete(M, pos_def, L_def, r_def):
    """
    Vytvoří matici kompletní struktury N+1 buněk s jednou poruchou na místě pos_def.
    """
    M_one_cell = matrix_one_cell(M)
    M_def = matrix_defect(M, L_def, r_def)
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
    S0_val = sp.pi * r0_val**2
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

def R_one_cell(s):
    s = np.asarray(s)
    cond1 = s < (1/4 * L_val)
    cond2 = (s >= (1/4 * L_val)) & (s < (3/4 * L_val))
    cond3 = s >= (3/4 * L_val)
    return np.select([cond1, cond2, cond3], [rA_val, rB_val, rA_val], default=r0_val)

def R_complete(s):
    """
    Vytvoří profil celé struktury:
    - buňky před poruchou (pos_def - 1)
    - porucha o délce L_def a poloměru r_def
    - zbylé buňky za poruchou (N - pos_def + 1)
    """
    s = np.asarray(s)
    R_vals = np.zeros_like(s, dtype=float)
    
    # Výpočet rozložení
    cells_before = pos_def - 1
    len_before = cells_before * L_val
    
    # 1. Část před poruchou
    mask_before = s < len_before
    # Modulo zajistí periodické opakování jedné buňky na požadované délce
    R_vals[mask_before] = R_one_cell(s[mask_before] % L_val)
    
    # 2. Porucha
    mask_def = (s >= len_before) & (s < len_before + L_def)
    R_vals[mask_def] = r_def
    
    # 3. Část za poruchou
    mask_after = s >= (len_before + L_def)
    # Od hodnoty 's' odečteme délku předchozích částí, aby modulo opět počítalo od nuly
    R_vals[mask_after] = R_one_cell((s[mask_after] - (len_before + L_def)) % L_val)
    
    return R_vals

def plot_R_complete(pos_def, L_def, r_def):
    # Celková délka odpovídá N normálním buňkám a 1 poruše
    total_length = (N * L_val) + L_def
    
    s_vals = np.linspace(0, total_length, 8000)
    R_vals = R_complete(s_vals)

    plt.figure(figsize=(12, 4))
    plt.plot(s_vals, R_vals, color='darkred')
    plt.title(f"Profil kompletní struktury (N={N} buněk, porucha na pozici {pos_def})")
    plt.xlabel("s")
    plt.ylabel("R", rotation=0)
    plt.xlim(0, total_length)
    
    # Dynamické nastavení osy Y podle toho, který poloměr je největší
    plt.ylim(0, max(rB_val, r_def) + 0.5)
    plt.grid(True)
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
# pozice buňky s poruchou
pos_def = 3
# délka buňky s poruchou
L_def = 1
# poloměr buňky s poruchou
r_def = 3

plot_R_complete(pos_def, L_def, r_def)

M_complete = matrix_complete(M, pos_def, L_def, r_def)
t_complete = get_transmission(M_complete)
plot_transmission(t_complete, 0, 5)