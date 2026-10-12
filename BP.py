import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as spi
# from numpy.linalg import matrix_power

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
    M_cell = M1 * M2 * M1
    return M_cell

def matrix_power_chebyshev(M, n):
    """
    Vypočítá n-tou mocninu 2x2 matice M (předpokládá se det(M) = 1)
    pomocí Čebyševových polynomů.
    """
    if n == 0:
        return sp.eye(2)
    if n == 1:
        return M  
    
    # Polovina stopy matice 
    x = (M[0, 0] + M[1, 1]) / 2

    # Rekurzivní generátor Čebyševova polynomu
    def U(m):
        if m < 0: return sp.S(0)
        if m == 0: return sp.S(1)
        u0, u1 = sp.S(1), 2 * x
        for _ in range(1, m):
            u0, u1 = u1, 2 * x * u1 - u0
        return u1
    
    # Výpočet koeficientů
    U_n_minus_1 = U(n - 1)
    U_n_minus_2 = U(n - 2)

    # Cayley-Hamilton
    M_n = U_n_minus_1 * M - U_n_minus_2 * sp.eye(2)

    return sp.Matrix(M_n)

def matrix_complete(M, pos_def1, L_def1, r_def1, pos_def2, L_def2, r_def2):
    """
    Vytvoří matici kompletní struktury s N normálními buňkami a 2 poruchami.
    pos_def1: pozice 1. poruchy
    pos_def2: pozice 2. poruchy (musí být > pos_def1)
    """
    M_one_cell = matrix_one_cell(M)
    M_def1 = substitute_matrix(M, L_def1, r_def1)
    M_def2 = substitute_matrix(M, L_def2, r_def2)
    
    # Počty normálních buněk v jednotlivých úsecích
    cells_before_1 = pos_def1 - 1
    cells_between = pos_def2 - pos_def1
    cells_after_2 = N - pos_def2 + 1
    
    # Skládání matic
    M_comp = (matrix_power_chebyshev(M_one_cell, cells_before_1) @ 
              M_def1 @ 
              matrix_power_chebyshev(M_one_cell, cells_between) @ 
              M_def2 @ 
              matrix_power_chebyshev(M_one_cell, cells_after_2))
    
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

def plot_transmission(t, min, max, root1=None, root2=None):
    k_vals = np.linspace(min, max, 1000)
    t_vals = t(k_vals)
    t_abs = np.abs(t_vals)
    plt.figure(figsize=(10, 5))
    plt.plot(k_vals, t_abs, color='darkred', label='|T|')
    if root1 is not None and root2 is not None:
        plt.axvspan(root1, root2, color='navy', alpha=0.1)
        plt.axvspan(1.05 * root1, 0.95 * root2, color='darkred', alpha=0.2)
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

def R_complete(s, pos_def1, L_def1, r_def1, pos_def2, L_def2, r_def2):
    """
    Vytvoří profil celé struktury se 2 poruchami.
    """
    s = np.asarray(s)
    R_vals = np.zeros_like(s, dtype=float)
    
    # Počty normálních buněk v jednotlivých úsecích
    cells_before_1 = pos_def1 - 1
    cells_between = pos_def2 - pos_def1
    
    # Délky k jednotlivým zlomům
    len1 = cells_before_1 * L_val                # konec úseku před 1. poruchou
    len2 = len1 + L_def1                         # konec 1. poruchy
    len3 = len2 + (cells_between * L_val)        # konec úseku mezi poruchami
    len4 = len3 + L_def2                         # konec 2. poruchy
    
    # 1. Část před 1. poruchou
    mask1 = s < len1
    R_vals[mask1] = R_one_cell(s[mask1] % L_val)
    
    # 2. První porucha
    mask2 = (s >= len1) & (s < len2)
    R_vals[mask2] = r_def1
    
    # 3. Část mezi poruchami
    mask3 = (s >= len2) & (s < len3)
    R_vals[mask3] = R_one_cell((s[mask3] - len2) % L_val)
    
    # 4. Druhá porucha
    mask4 = (s >= len3) & (s < len4)
    R_vals[mask4] = r_def2
    
    # 5. Část za 2. poruchou
    mask5 = s >= len4
    R_vals[mask5] = R_one_cell((s[mask5] - len4) % L_val)
    
    return R_vals

def plot_R_complete(pos_def1, L_def1, r_def1, pos_def2, L_def2, r_def2):
    # Celková délka odpovídá N normálním buňkám a 2 poruchám
    total_length = (N * L_val) + L_def1 + L_def2
    
    s_vals = np.linspace(0, total_length, 8000)
    R_vals = R_complete(s_vals, pos_def1, L_def1, r_def1, pos_def2, L_def2, r_def2)

    # Výpočet zlomových bodů pro určení polohy poruch
    cells_before_1 = pos_def1 - 1
    cells_between = pos_def2 - pos_def1
    
    len1 = cells_before_1 * L_val                # začátek 1. poruchy
    len2 = len1 + L_def1                         # konec 1. poruchy
    len3 = len2 + (cells_between * L_val)        # začátek 2. poruchy
    len4 = len3 + L_def2                         # konec 2. poruchy

    plt.figure(figsize=(12, 4))
    
    # Vykreslení celé struktury jako základu (černě)
    plt.plot(s_vals, R_vals, color='black', linewidth=1.5, label='Normální buňky')
    
    # Barevné překreslení poruch
    mask_def1 = (s_vals >= len1) & (s_vals <= len2)
    mask_def2 = (s_vals >= len3) & (s_vals <= len4)
    
    plt.plot(s_vals[mask_def1], R_vals[mask_def1], color='darkred', linewidth=3, label='1. porucha')
    plt.plot(s_vals[mask_def2], R_vals[mask_def2], color='navy', linewidth=3, label='2. porucha')
    
    plt.fill_between(s_vals[mask_def1], 0, R_vals[mask_def1], color='darkred', alpha=0.1)
    plt.fill_between(s_vals[mask_def2], 0, R_vals[mask_def2], color='navy', alpha=0.1)

    plt.title(f"Profil kompletní struktury (N={N} buněk, poruchy na {pos_def1} a {pos_def2})")
    plt.xlabel("s")
    plt.ylabel("R", rotation=0)
    plt.xlim(0, total_length)
    
    # Dynamické nastavení osy Y
    plt.ylim(0, max(rB_val, r_def1, r_def2) + 0.5)
    plt.grid(True)
    plt.legend()
    plt.show()

def get_xi(M):
    """
    Z matice jedné buňky vrátí poloviční stopu (xi = cos(qL)) jako funkci k.
    """
    xi_sym = (M[0, 0] + M[1, 1]) / 2
    xi_sub = xi_sym.subs({Z0: Z0_val})

    # Převod na numerickou funkci
    return sp.lambdify(k, xi_sub, modules='numpy')

def plot_bloch_phase(xi_func, min_k, max_k):
    """
    Vykreslí závislost poloviční stopy na k.
    """
    k_vals = np.linspace(min_k, max_k, 1000)
    
    # Lambdify může u symbolických výrazů vracet komplexní čísla s nulovou imag. částí
    xi_vals = np.real(xi_func(k_vals))
    
    plt.figure(figsize=(12, 6))
    plt.plot(k_vals, xi_vals, color = 'darkred', label=r'$\xi(k)$')
    
    plt.axhline(1, color='navy')
    plt.axhline(-1, color='navy')
    
    plt.title("Blochova fáze")
    plt.xlabel("k")
    plt.ylabel(r"$\xi = \frac{1}{2}\mathrm{Tr}(M)$", rotation = 0)
    plt.xlim(min_k, max_k)
    plt.ylim(-2.5, 1.5)
    plt.grid(True)
    plt.show()

def find_root_bisection(xi_func, k_min, k_max, target=-1):
    """
    Najde kořen rovnice xi(k) = target pomocí metody půlení intervalů.
    """
    f = lambda k_val: np.real(xi_func(k_val)) - target
    # Použití bisekce se zadanou tolerancí
    res = opt.root_scalar(f, bracket=[k_min, k_max], method='bisect', xtol=1e-12)
    return res.root


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
# t_one_cell = get_transmission(M_one_cell)
# plot_transmission(t_one_cell, 0, 10)

# Výpočet a vykreslení Blochovy fáze
xi_function = get_xi(M_one_cell)
plot_bloch_phase(xi_function, 0, 10)

# Hledání kořenů (interval pro band gap)
root1 = find_root_bisection(xi_function, 1.5, 2.3, target=-1)
print(f"Kořen 1: {root1}")
root2 = find_root_bisection(xi_function, 4.1, 4.8, target=-1)
print(f"Kořen 2: {root2}")

# přenos pro N buněk jako funkce k
M_n_cells = matrix_power_chebyshev(M_one_cell, N)
t_n_cells = get_transmission(M_n_cells)
plot_transmission(t_n_cells, 0, 5, root1, root2)

# přenos pro N+2 buněk se dvěma poruchami jako funkce k
# 1. porucha
pos_def1 = 5   # od 1 do 11
L_def1 = 1
r_def1 = 1

# 2. porucha
pos_def2 = 7  # (musí být >= pos_def1)
L_def2 = 1.5
r_def2 = 0.5 

# Vykreslení profilu struktury
plot_R_complete(pos_def1, L_def1, r_def1, pos_def2, L_def2, r_def2)

# Výpočet a vykreslení přenosu
M_complete = matrix_complete(M, pos_def1, L_def1, r_def1, pos_def2, L_def2, r_def2)
t_complete = get_transmission(M_complete)
plot_transmission(t_complete, 0, 5, root1, root2)

# Výpočet integrálu v band gapu
func_abs_t = lambda k_val: np.abs(t_complete(k_val))
integral_scipy, error = spi.quad(func_abs_t, 1.05*root1, 0.95*root2, limit=200)
print(f"Integrál (SciPy): {integral_scipy}, odhad chyby: {error}")