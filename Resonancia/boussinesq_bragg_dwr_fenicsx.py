"""
boussinesq_bragg_dwr_fenicsx.py
=========================================================================
Resonancia de Bragg — Chen-Boussinesq 2D con FONDO VARIABLE
Método de Rothe (Crank-Nicolson) + DWR goal-oriented — dolfinx 0.10.x

Versión goal-oriented (DWR) del solver de Bragg: reemplaza el estimador
residual de energía por el control de error dual-weighted residual de
Rognes-Logg, reimplementado a mano (no existe en dolfinx). El problema
dual se deriva AUTOMÁTICAMENTE con derivative(), incluidos los términos
de batimetría — no hay que derivar nada del fondo a mano.

CAMBIOS respecto a la versión anterior:
  - Funcional objetivo M: mareógrafo numérico REGULARIZADO (gauge gaussiano)
    en la zona de reflexión, en lugar de M = ∫η² global. La onda reflejada
    que mide el gauge existe SOLO por el forzamiento del fondo: el funcional
    apunta directamente a la señal generada por la batimetría, y la
    adaptatividad concentra el esfuerzo justo ahí.
  - Se guarda el ESTADO INICIAL (t=0) antes del bucle temporal: el primer
    snapshot ya no aparece "iniciado".
  - Serie temporal M(t) registrada en CSV para las figuras del artículo.

Batimetría de Bragg (estática):
    h(y) = -0.5·sin(0.4·(y-100))   si 100 ≤ y ≤ 179,   0 en otro caso
    ∂h/∂t = 0,  ∂²h/∂t² = 0   (estructura lista para fondo móvil)

Coste: el dual en P2 + re-solver en cada refinamiento es caro para 500
pasos. ADAPT_EVERY controla cada cuántos pasos se estima/refina; entre
medias se reutiliza la malla (solo primal).

Requiere: dolfinx 0.10.x, basix, ufl, petsc4py, mpi4py, numpy.
=========================================================================
"""

import csv
import numpy as np
from math import sqrt
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.io
from dolfinx.mesh import (create_rectangle, CellType,
                          locate_entities_boundary, refine)
from dolfinx.fem import (functionspace, Function, Constant,
                         dirichletbc, locate_dofs_topological,
                         assemble_scalar, form)
from dolfinx.fem.petsc import (assemble_vector, NonlinearProblem,
                               LinearProblem)
from basix.ufl import element as basix_element, mixed_element

import ufl
from ufl import dx, grad, inner, split, TestFunctions, TestFunction

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
T          = 500.0
NUM_STEPS  = 500
DT         = T / NUM_STEPS               # = 1.0
THETA_CN   = 0.5

# Coeficiente del forzamiento (∇h)_tt en las ecuaciones de momento.
# Valor confirmado por el autor (mejia2021analytical): γ = 1 + √(2/3).
# OJO de consistencia: el borrador del .tex define en la Sec. 3 γ = 1 - √(2/3);
# ese signo del .tex debe corregirse a 1 + √(2/3) para que código y artículo
# coincidan. Término INERTE mientras el fondo sea estático (h_tt = 0).
KK_VAL     = 1.0 + sqrt(2.0 / 3.0)

THETA_DORF = 0.5
TOL_DWR    = 1.0e-6
MAX_ADAPT  = 5
ADAPT_EVERY = 10        # estimar/refinar cada N pasos (1 = siempre)

NX, NY     = 30, 500    # NY reducido para demo (sube para producción)
SAVE_EVERY = 10

# ---- Funcional objetivo: estación de monitoreo (mareógrafo numérico) ------
# Estación en la zona de reflexión [-200,-50] donde se forma la onda de Bragg.
# Esa onda reflejada existe SOLO por el fondo variable -> el gauge apunta
# directamente al fenómeno de interés. σ ≈ media long. de onda reflejada (≈32.7).
GAUGE_X0    = 0.0
GAUGE_Y0    = -100.0
GAUGE_SIGMA = 5.0

comm = MPI.COMM_WORLD


# ===========================================================================
# RESIDUO PARAMÉTRICO con BATIMETRÍA
# ===========================================================================

def chen_residual(u, u_n, bath, tests, k, theta=THETA_CN):
    """Residuo Crank-Nicolson de Chen con fondo variable.

    bath = (h_n, h_n1, h_t, h_t1, h_tt, h_tt1, p0)
    Dispersivo correcto: +(1/6)/k (∇(u-u_n), ∇v).
    """
    v1, v2, v3 = tests
    u1, u2, u3 = split(u)
    n1, n2, n3 = split(u_n)
    h_n, h_n1, h_t, h_t1, h_tt, h_tt1, p0 = bath
    tt  = theta
    ott = 1.0 - theta
    KK  = Constant(u.function_space.mesh, PETSc.ScalarType(KK_VAL))

    # ---- u1 ----
    F1 = (
        ((u1 - n1) / k) * v1 * dx
        + (1.0/6.0) / k * inner(grad(u1 - n1), grad(v1)) * dx
        - tt  * u3 * grad(v1)[0] * dx
        - ott * n3 * grad(v1)[0] * dx
        - (tt  / 2.0) * (u1**2 + u2**2) * grad(v1)[0] * dx
        - (ott / 2.0) * (n1**2 + n2**2) * grad(v1)[0] * dx
        - tt  * p0 * grad(v1)[0] * dx
        - ott * p0 * grad(v1)[0] * dx
        + tt  * KK * h_tt  * grad(v1)[0] * dx
        + ott * KK * h_tt1 * grad(v1)[0] * dx
    )
    # ---- u2 ----
    F2 = (
        ((u2 - n2) / k) * v2 * dx
        + (1.0/6.0) / k * inner(grad(u2 - n2), grad(v2)) * dx
        - tt  * u3 * grad(v2)[1] * dx
        - ott * n3 * grad(v2)[1] * dx
        - (tt  / 2.0) * (u1**2 + u2**2) * grad(v2)[1] * dx
        - (ott / 2.0) * (n1**2 + n2**2) * grad(v2)[1] * dx
        - tt  * p0 * grad(v2)[1] * dx
        - ott * p0 * grad(v2)[1] * dx
        + tt  * KK * h_tt  * grad(v2)[1] * dx
        + ott * KK * h_tt1 * grad(v2)[1] * dx
    )
    # ---- η (u3) ----
    F3 = (
        ((u3 - n3) / k) * v3 * dx
        + (1.0/6.0) / k * inner(grad(u3 - n3), grad(v3)) * dx
        - tt  * u1 * grad(v3)[0] * dx
        - ott * n1 * grad(v3)[0] * dx
        - tt  * u2 * grad(v3)[1] * dx
        - ott * n2 * grad(v3)[1] * dx
        - tt  * u3 * u1 * grad(v3)[0] * dx
        - ott * n3 * n1 * grad(v3)[0] * dx
        - tt  * u3 * u2 * grad(v3)[1] * dx
        - ott * n3 * n2 * grad(v3)[1] * dx
        # términos de fondo h·u
        - tt  * h_n  * u1 * grad(v3)[0] * dx
        - ott * h_n1 * n1 * grad(v3)[0] * dx
        - tt  * h_n  * u2 * grad(v3)[1] * dx
        - ott * h_n1 * n2 * grad(v3)[1] * dx
        # derivada temporal del fondo ∂h/∂t
        + tt  * h_t  * v3 * dx
        + ott * h_t1 * v3 * dx
        # >>> OPCIONAL: término γ̃·Δh del esquema discreto del artículo (ec. cn:eta).
        # >>> Ausente en tu código previo; reconcilia con la disertación si lo necesitas:
        # gtil = (1./3.)*(2.-sqrt(6.))
        # - tt*gtil*inner(grad(h_n), grad(v3))*dx - ott*gtil*inner(grad(h_n1), grad(v3))*dx
    )
    return F1 + F2 + F3


def goal_functional(u):
    """
    Mareógrafo numérico regularizado:
        M(U) = ∫ η(x) ψ_σ(x - x0) dx,
    con ψ_σ gaussiano normalizado (∫ψ_σ ≈ 1). Lineal en η:
        M'(U)[Φ] = ∫ ψ_σ φ3 dx   (fuente del dual solo en la componente η).

    En 2D la evaluación puntual pura NO es acotada sobre H¹ (Sobolev s>n/2=1);
    el núcleo gaussiano la regulariza y deja el dual bien planteado.
    """
    mesh = u.function_space.mesh
    x = ufl.SpatialCoordinate(mesh)
    _, _, u3 = split(u)
    norm = 1.0 / (2.0 * ufl.pi * GAUGE_SIGMA**2)
    psi  = norm * ufl.exp(-((x[0]-GAUGE_X0)**2 + (x[1]-GAUGE_Y0)**2)
                          / (2.0 * GAUGE_SIGMA**2))
    # cuadratura alta: el núcleo gaussiano no es polinómico y en celdas gruesas
    # la regla por defecto lo sub-integra, contaminando la tasa de convergencia.
    dx_q = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 6})
    return u3 * psi * dx_q


def gauge_value(u_h):
    """Valor escalar del funcional objetivo M(U_h) (suma global)."""
    val = assemble_scalar(form(goal_functional(u_h)))
    return comm.allreduce(val, op=MPI.SUM)


# ===========================================================================
# Malla, espacios, BC, batimetría, dato inicial
# ===========================================================================

def make_mesh():
    return create_rectangle(
        comm, [np.array([-15.0, -400.0]), np.array([15.0, 600.0])],
        [NX, NY], CellType.triangle)


def make_space(mesh, degree=1):
    Pk = basix_element("Lagrange", mesh.basix_cell(), degree)
    return functionspace(mesh, mixed_element([Pk, Pk, Pk]))


def make_bcs(V, mesh):
    fdim = mesh.topology.dim - 1
    def inflow(x):  return np.isclose(x[1], -400.0)
    def outflow(x): return np.isclose(x[1],  600.0)
    def wall1(x):   return np.isclose(x[0], -15.0)
    def wall2(x):   return np.isclose(x[0],  15.0)
    V0, _ = V.sub(0).collapse(); V1, _ = V.sub(1).collapse()
    z0 = Function(V0); z0.x.array[:] = 0.0
    z1 = Function(V1); z1.x.array[:] = 0.0
    bcs = []
    for m in [inflow, outflow, wall1, wall2]:
        fc = locate_entities_boundary(mesh, fdim, m)
        bcs.append(dirichletbc(z0, locate_dofs_topological((V.sub(0), V0), fdim, fc), V.sub(0)))
        bcs.append(dirichletbc(z1, locate_dofs_topological((V.sub(1), V1), fdim, fc), V.sub(1)))
    return bcs


def h_bragg(x):
    y = x[1]
    return np.where((y >= 100.0) & (y <= 179.0),
                    -0.5 * np.sin(0.4 * (y - 100.0)), 0.0)


def make_bathymetry(W):
    """7 funciones escalares en W=P1. Fondo estático: h_t = h_tt = 0."""
    h_n   = Function(W); h_n.interpolate(h_bragg)
    h_n1  = Function(W); h_n1.interpolate(h_bragg)
    zeros = [Function(W) for _ in range(5)]
    for z in zeros:
        z.x.array[:] = 0.0
    h_t, h_t1, h_tt, h_tt1, p0 = zeros
    return (h_n, h_n1, h_t, h_t1, h_tt, h_tt1, p0)


def initial_condition(V):
    u_n = Function(V)
    V0, d0 = V.sub(0).collapse(); V1, d1 = V.sub(1).collapse(); V2, d2 = V.sub(2).collapse()
    u_n.x.array[d0] = 0.0
    g = Function(V1); g.interpolate(lambda x: 0.01 * np.exp(-x[1]**2 / 60.0))
    u_n.x.array[d1] = g.x.array
    e = Function(V2); e.interpolate(lambda x: 0.01 * np.exp(-x[1]**2 / 60.0))
    u_n.x.array[d2] = e.x.array
    return u_n


# ===========================================================================
# Interpolación mixta / escalar entre mallas y grados
# ===========================================================================

def interp_mixed(V_to, u_from):
    u_to = Function(V_to)
    same = V_to.mesh is u_from.function_space.mesh
    if not same:
        from dolfinx.fem import create_interpolation_data
        tdim  = V_to.mesh.topology.dim
        cells = np.arange(V_to.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    for i in range(3):
        Vi_to, dofs_to = V_to.sub(i).collapse()
        Vi_fr, dofs_fr = u_from.function_space.sub(i).collapse()
        a = Function(Vi_fr); a.x.array[:] = u_from.x.array[dofs_fr]
        b = Function(Vi_to)
        if same:
            b.interpolate(a)
        else:
            from dolfinx.fem import create_interpolation_data
            nmm = create_interpolation_data(Vi_to, Vi_fr, cells, padding=1e-6)
            b.interpolate_nonmatching(a, cells, nmm)
        u_to.x.array[dofs_to] = b.x.array
    return u_to


def interp_scalar(W_to, f_from):
    same = W_to.mesh is f_from.function_space.mesh
    g = Function(W_to)
    if same:
        g.interpolate(f_from)
    else:
        from dolfinx.fem import create_interpolation_data
        tdim  = W_to.mesh.topology.dim
        cells = np.arange(W_to.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        nmm = create_interpolation_data(W_to, f_from.function_space, cells, padding=1e-6)
        g.interpolate_nonmatching(f_from, cells, nmm)
    return g


# ===========================================================================
# Step 2 — primal
# ===========================================================================

def solve_primal(mesh, u_n_prev, bath_prev):
    V = make_space(mesh, 1)
    W = functionspace(mesh, ("Lagrange", 1))
    bcs = make_bcs(V, mesh)
    u_n  = interp_mixed(V, u_n_prev)
    bath = tuple(interp_scalar(W, f) for f in bath_prev)
    u = Function(V); u.x.array[:] = u_n.x.array
    k = Constant(mesh, PETSc.ScalarType(DT))
    F = chen_residual(u, u_n, bath, TestFunctions(V), k)
    prob = NonlinearProblem(
        F, u, bcs=bcs, petsc_options_prefix="bragg_primal_",
        petsc_options={"snes_type": "newtonls", "snes_rtol": 1e-10,
                       "snes_atol": 1e-12, "snes_max_it": 50,
                       "ksp_type": "gmres", "ksp_gmres_restart": 100,
                       "ksp_rtol": 1e-12,
                       "pc_type": "bjacobi", "sub_pc_type": "ilu",
                       "snes_linesearch_type": "basic"})
    prob.solve()
    return u, V, u_n, bath


# ===========================================================================
# Steps 3+4 — dual en P2 y peso w = z2 - π_h z2
# ===========================================================================

# ===========================================================================
# Extrapolación P1 -> P2 del dual (Rognes-Logg, Algoritmo 1), reimplementada
# por mínimos cuadrados móviles sobre parches de vecinos. Sustituye el solve
# directo del dual en P2: el peso w = E(z_h) - z_h aproxima z - π_h z con la
# magnitud correcta, lo que corrige el índice de efectividad.
# Nota: pensada para ejecución en SERIE (usa coords de dofs locales).
# ===========================================================================

def _mls_extrapolate(Xc, valc, Xf, k=12, reg=1e-10):
    """Ajuste cuadrático local por mínimos cuadrados, centrado en cada punto de
    Xf usando sus k vecinos más cercanos en Xc. Devuelve el valor recuperado en
    Xf (el término constante del ajuste centrado)."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as e:
        raise ImportError("La extrapolación del dual requiere scipy "
                          "(pip install scipy).") from e
    kk = int(min(k, Xc.shape[0]))
    tree = cKDTree(Xc)
    _, idx = tree.query(Xf, k=kk)
    if idx.ndim == 1:
        idx = idx[:, None]
    P  = Xc[idx]                                   # (n2, kk, 2)
    dx = P[:, :, 0] - Xf[:, None, 0]
    dy = P[:, :, 1] - Xf[:, None, 1]
    one = np.ones_like(dx)
    A = np.stack([one, dx, dy, dx*dx, dx*dy, dy*dy], axis=2)   # base cuadrática
    b = valc[idx]                                  # (n2, kk)
    AtA = np.einsum('nki,nkj->nij', A, A)
    Atb = np.einsum('nki,nk->ni', A, b)
    tr  = np.einsum('nii->n', AtA) / 6.0
    AtA += (reg * np.maximum(tr, 1e-30))[:, None, None] * np.eye(6)[None, :, :]
    c = np.linalg.solve(AtA, Atb)                  # (n2, 6)
    return c[:, 0]


def extrapolate_mixed_p1_to_p2(z1, V2):
    """Extrapola un dual P1 mixto (3 componentes) a P2, componente por componente."""
    Ez = Function(V2)
    for i in range(3):
        Vi1, d1 = z1.function_space.sub(i).collapse()
        Vi2, d2 = V2.sub(i).collapse()
        Xc = Vi1.tabulate_dof_coordinates()[:, :2]
        Xf = Vi2.tabulate_dof_coordinates()[:, :2]
        Ez.x.array[d2] = _mls_extrapolate(Xc, z1.x.array[d1], Xf)
    Ez.x.scatter_forward()
    return Ez


def solve_dual_and_weight(mesh, u_h, u_n, bath):
    # Dual en el MISMO espacio que el primal (P1), siguiendo Rognes-Logg.
    # La batimetría entra en el dual automáticamente vía derivative().
    V1   = u_h.function_space
    bcs1 = make_bcs(V1, mesh)
    k = Constant(mesh, PETSc.ScalarType(DT))
    F1     = chen_residual(u_h, u_n, bath, TestFunctions(V1), k)
    a_dual = ufl.adjoint(ufl.derivative(F1, u_h))
    L_dual = ufl.derivative(goal_functional(u_h), u_h)
    prob = LinearProblem(a_dual, L_dual, bcs=bcs1,
                         petsc_options_prefix="bragg_dual_",
                         petsc_options={"ksp_type": "gmres", "ksp_gmres_restart": 100,
                                        "pc_type": "bjacobi", "sub_pc_type": "ilu",
                                        "ksp_rtol": 1e-12})
    z1 = prob.solve()
    # Peso w = E(z1) - z1  (extrapolado P2 menos dual P1 elevado a P2)
    V2   = make_space(mesh, 2)
    Ez   = extrapolate_mixed_p1_to_p2(z1, V2)
    z1_2 = interp_mixed(V2, z1)
    w = Function(V2); w.x.array[:] = Ez.x.array - z1_2.x.array
    return w


# ===========================================================================
# Step 5 — estimador global e indicadores locales
# ===========================================================================

def dwr_estimate(mesh, u_h, u_n, bath, w):
    k = Constant(mesh, PETSc.ScalarType(DT))
    w1, w2, w3 = split(w)
    # --- estimador global ---
    eta_g = assemble_scalar(form(chen_residual(u_h, u_n, bath, (w1, w2, w3), k)))
    eta_global = abs(comm.allreduce(eta_g, op=MPI.SUM))
    # --- localización por partición de unidad (sombreros P1, Richter-Wick) ---
    # Más robusta que DG0: los sombreros continuos φ_i (Σφ_i = 1) evitan la
    # cancelación espuria de facetas del indicador DG0, que marcaba mal.
    P1  = functionspace(mesh, ("Lagrange", 1))
    phi = TestFunction(P1)
    F_loc = chen_residual(u_h, u_n, bath, (w1*phi, w2*phi, w3*phi), k)
    vec = assemble_vector(form(F_loc))
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD,    mode=PETSc.ScatterMode.REVERSE)
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    eta_vert = np.abs(vec.array)                    # indicador por vértice
    # vértice -> celda: eta_K = suma de los indicadores de los vértices de K
    tdim = mesh.topology.dim
    nloc = mesh.topology.index_map(tdim).size_local
    cell_dofs = P1.dofmap.list                      # (ncells, 3) dofs P1 por celda
    eta_K = eta_vert[cell_dofs[:nloc]].sum(axis=1)
    return eta_global, eta_K


# ===========================================================================
# Step 7 — Dörfler + refinamiento
# ===========================================================================

def dorfler_mark(eta_K, theta):
    gsum = comm.allreduce(float(np.sum(eta_K**2)), op=MPI.SUM)
    idx  = np.argsort(eta_K)[::-1]
    target, cumsum, marked = theta*gsum, 0.0, []
    for i in idx:
        if cumsum >= target:
            break
        marked.append(i); cumsum += eta_K[i]**2
    return np.array(marked, dtype=np.int32)


def refine_mesh(mesh, marked):
    mesh.topology.create_entities(1)
    mesh.topology.create_connectivity(2, 1)
    c2e = mesh.topology.connectivity(2, 1)
    edges = set()
    for c in marked:
        for e in c2e.links(c):
            edges.add(int(e))
    new_mesh, _, _ = refine(mesh, np.array(sorted(edges), dtype=np.int32))
    new_mesh.topology.create_connectivity(new_mesh.topology.dim,
                                          new_mesh.topology.dim - 1)
    return new_mesh


def save_step(u, mesh, t, step):
    u1, u2, eta = u.split()
    u1.name, u2.name, eta.name = "u1", "u2", "eta"
    s = f"{step:04d}"
    for fn, tag in ((u1, "u1"), (u2, "u2"), (eta, "eta")):
        with dolfinx.io.XDMFFile(comm, f"braggdwr_{tag}_t{s}.xdmf", "w") as xf:
            xf.write_mesh(mesh); xf.write_function(fn, t)


# ===========================================================================
# Loop principal
# ===========================================================================

def main():
    if comm.rank == 0:
        print("=" * 70)
        print("  Resonancia de Bragg — DWR goal-oriented — dolfinx 0.10")
        print(f"  M=gauge({GAUGE_X0},{GAUGE_Y0}) σ={GAUGE_SIGMA} | dual P2 | "
              f"Dörfler θ={THETA_DORF} | adaptar c/{ADAPT_EVERY} pasos")
        print("=" * 70)

    mesh = make_mesh()
    V    = make_space(mesh, 1)
    W    = functionspace(mesh, ("Lagrange", 1))
    u_n  = initial_condition(V)
    bath = make_bathymetry(W)

    # --- guardar el ESTADO INICIAL (t=0) ANTES de evolucionar ---
    save_step(u_n, mesh, 0.0, 0)
    gauge_log = [(0, 0.0, mesh.topology.index_map(2).size_global,
                  gauge_value(u_n), float("nan"))]
    if comm.rank == 0:
        print(f"  paso    0: t=0.0 (dato inicial) | "
              f"{mesh.topology.index_map(2).size_global} celdas")

    t = 0.0
    for step in range(NUM_STEPS):
        t += DT
        do_adapt = (step % ADAPT_EVERY == 0)
        eta_log = float("nan")

        if do_adapt:
            for adapt_it in range(MAX_ADAPT):
                n_cells = mesh.topology.index_map(2).size_global
                u_h, V_c, u_n_c, bath_c = solve_primal(mesh, u_n, bath)
                w = solve_dual_and_weight(mesh, u_h, u_n_c, bath_c)
                eta_global, eta_K = dwr_estimate(mesh, u_h, u_n_c, bath_c, w)
                eta_log = eta_global
                if comm.rank == 0:
                    print(f"  paso {step+1:3d} [adapt {adapt_it+1}/{MAX_ADAPT}] "
                          f"celdas={n_cells:7d} | η_DWR={eta_global:.4e}")
                if eta_global <= TOL_DWR:
                    break
                if adapt_it < MAX_ADAPT - 1:
                    mesh = refine_mesh(mesh, dorfler_mark(eta_K, THETA_DORF))
        else:
            # paso barato: solo primal en la malla actual
            u_h, V_c, u_n_c, bath_c = solve_primal(mesh, u_n, bath)

        Mval = gauge_value(u_h)
        ncel = mesh.topology.index_map(2).size_global
        gauge_log.append((step + 1, t, ncel, Mval, eta_log))

        if (step + 1) % SAVE_EVERY == 0:
            save_step(u_h, mesh, t, step + 1)
            if comm.rank == 0:
                print(f"  paso {step+1}: t={t:.1f} guardado | M={Mval:.6e} | "
                      f"{ncel} celdas")

        # avanzar: re-interpolar a la malla actual si cambió
        if u_h.function_space.mesh is not mesh:
            V_new = make_space(mesh, 1); W_new = functionspace(mesh, ("Lagrange", 1))
            u_n  = interp_mixed(V_new, u_h)
            bath = tuple(interp_scalar(W_new, f) for f in bath_c)
        else:
            u_n, bath = u_h, bath_c

    # --- serie temporal del gauge para las figuras del artículo ---
    if comm.rank == 0:
        with open("bragg_gauge.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["step", "t", "ncells", "M_gauge", "eta_dwr"])
            wr.writerows(gauge_log)
        print("\n" + "=" * 70)
        print(f"  Completa. braggdwr_*_t####.xdmf (cada {SAVE_EVERY} pasos) | "
              f"serie del gauge en bragg_gauge.csv")
        print("=" * 70)


if __name__ == "__main__":
    main()
