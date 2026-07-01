"""
dwr_convergence.py  (v2 — mallas ANIDADAS, datos consistentes)
=========================================================================
Estudio de convergencia UNIFORME vs ADAPTATIVO del funcional objetivo M en un
paso de Rothe representativo (t*), reutilizando las funciones ya verificadas de
los solvers (boussinesq_cilindro / boussinesq_bragg).

POR QUE ANIDADAS
----------------
La version anterior comparaba contra mallas gmsh INDEPENDIENTES (no anidadas):
u_star se interpolaba distinto en cada una, y esa diferencia de representacion
del dato contaminaba 'err' (sin entrar en eta), dando I_eff espurio y pendientes
falsas. El diagnostico dwr_effectivity.py confirmo que con datos CONSISTENTES
I_eff -> 1 y la tasa uniforme es O(h^2).

Aqui se usa esa misma receta:
  - pre-roll en la malla BASE  -> u_n0 (en la base).
  - UNIFORME: refinamientos uniformes anidados de la base (1 triangulo -> 4).
  - ADAPTATIVO: ciclo DWR, que tambien refina de forma anidada desde la base.
  - u_n0 se transporta por anidamiento a todos los niveles (interpolacion
    EXACTA de una funcion P1 sobre cualquier subdivision).
  => 'err' mide solo discretizacion; coincide con lo que estima eta.

Produce:
  - convergencia_<exp>.csv / .png
  - malla_uniforme_<exp>.xdmf / malla_adaptativa_<exp>.xdmf

Uso:
    python dwr_convergence.py            # cilindro
    EXPERIMENT="bragg" python dwr_convergence.py
=========================================================================
"""

import os
import csv
import time
import numpy as np
from mpi4py import MPI

import dolfinx.io
from dolfinx.mesh import refine
from dolfinx.fem import functionspace

comm = MPI.COMM_WORLD

# ---------------------------------------------------------------------------
# Configuracion -- FIJADO al experimento de RESONANCIA DE BRAGG
# ---------------------------------------------------------------------------
EXPERIMENT = "bragg"

import boussinesq_bragg_dwr_fenicsx as ex
HAS_BATH       = True
PREROLL        = 280            # t* = 280: la onda reflejada de Bragg ya llego
                               # al gauge en (0,-100). Pre-roll largo -> costoso.
BASE           = (30, 500)     # base gruesa: refinar uniforme cuadruplica celdas
                               # (30x500 ~ 30k celdas -> ~120k -> ~480k en 3 niveles)
UNIFORM_LEVELS = 3             # niveles uniformes anidados (el ultimo = referencia)
ADAPT_ITERS    = 5             # iteraciones del ciclo adaptativo
SAVE_UNIF_LVL  = 1             # que nivel uniforme guardar para la figura de mallas

THETA_DORF = ex.THETA_DORF


# ---------------------------------------------------------------------------
# Adaptadores (ocultan con/sin batimetria) -- reutilizan funciones del solver
# ---------------------------------------------------------------------------

def build_base():
    if HAS_BATH:
        ex.NX, ex.NY = BASE
        return ex.make_mesh()
    return ex.make_mesh(BASE)


def fresh_bath(mesh):
    if not HAS_BATH:
        return None
    W = functionspace(mesh, ("Lagrange", 1))
    return ex.make_bathymetry(W)        # batimetria analitica, recreada en la malla


def step_full(mesh, u_n_prev, bath_prev):
    """Un paso primal. Devuelve (u_h, u_n_en_malla, bath_en_malla)."""
    if HAS_BATH:
        u_h, _, u_nc, bath_c = ex.solve_primal(mesh, u_n_prev, bath_prev)
        return u_h, u_nc, bath_c
    u_h, _, u_nc = ex.solve_primal(mesh, u_n_prev)
    return u_h, u_nc, None


def estimate(mesh, u_h, u_nc, bath_c):
    if HAS_BATH:
        w = ex.solve_dual_and_weight(mesh, u_h, u_nc, bath_c)
        return ex.dwr_estimate(mesh, u_h, u_nc, bath_c, w)
    w = ex.solve_dual_and_weight(mesh, u_h, u_nc)
    return ex.dwr_estimate(mesh, u_h, u_nc, w)


def ncells(mesh):
    return mesh.topology.index_map(2).size_global


def uniform_refine(mesh):
    """Refinamiento uniforme (rojo): marca TODAS las aristas -> 1 triangulo a 4."""
    mesh.topology.create_entities(1)
    ne = mesh.topology.index_map(1).size_local
    edges = np.arange(ne, dtype=np.int32)
    nm, _, _ = refine(mesh, edges)
    nm.topology.create_connectivity(nm.topology.dim, nm.topology.dim - 1)
    return nm


def save_mesh(mesh, name):
    with dolfinx.io.XDMFFile(comm, name, "w") as xf:
        xf.write_mesh(mesh)


# ---------------------------------------------------------------------------
# Pre-roll en la malla base (el dato sera consistente por anidamiento)
# ---------------------------------------------------------------------------

def preroll_base(base_mesh):
    if comm.rank == 0:
        print(f"[pre-roll] {PREROLL} pasos en malla base...")
    V = ex.make_space(base_mesh, 1)
    u_n = ex.initial_condition(V)
    bath = fresh_bath(base_mesh)
    for _ in range(PREROLL):
        u_h, _, bath_c = step_full(base_mesh, u_n, bath)
        u_n, bath = u_h, bath_c
    if comm.rank == 0:
        print(f"[pre-roll] listo. t*={PREROLL*ex.DT:.2f}, {ncells(base_mesh)} celdas | "
              f"M(u_n0)={ex.gauge_value(u_n):.6e}")
    return u_n  # u_n0 en la malla base


# ---------------------------------------------------------------------------
# Secuencia UNIFORME anidada
# ---------------------------------------------------------------------------

def run_uniform(u_n0, base_mesh, nlevels):
    rows, meshes = [], []
    mesh = base_mesh
    for L in range(nlevels):
        u_n  = ex.interp_mixed(ex.make_space(mesh, 1), u_n0)  # base->nivel L, EXACTO
        bath = fresh_bath(mesh)
        t0 = time.perf_counter()
        u_h, _, _ = step_full(mesh, u_n, bath)
        wall = time.perf_counter() - t0
        M, nc = ex.gauge_value(u_h), ncells(mesh)
        rows.append({"ncells": nc, "M": M, "wall": wall})
        meshes.append(mesh)
        if comm.rank == 0:
            print(f"  [uniforme L{L}] celdas={nc:8d} | M={M:.10e} | {wall:6.2f}s")
        if L < nlevels - 1:
            mesh = uniform_refine(mesh)
    return rows, meshes


# ---------------------------------------------------------------------------
# Ciclo ADAPTATIVO anidado (un paso de Rothe)
# ---------------------------------------------------------------------------

def run_adaptive(u_n0, base_mesh, M_ref, niters):
    rows = []
    mesh = base_mesh
    for it in range(niters):
        u_n  = ex.interp_mixed(ex.make_space(mesh, 1), u_n0)  # base->malla adapt, EXACTO
        bath = fresh_bath(mesh)
        t0 = time.perf_counter()
        u_h, u_nc, bath_c = step_full(mesh, u_n, bath)
        eta_g, eta_K = estimate(mesh, u_h, u_nc, bath_c)
        wall = time.perf_counter() - t0
        M   = ex.gauge_value(u_h)
        err = abs(M_ref - M)
        Ieff = (eta_g / err) if err > 0 else float("nan")
        nc  = ncells(mesh)
        rows.append({"ncells": nc, "M": M, "err": err, "eta": eta_g,
                     "Ieff": Ieff, "wall": wall})
        if comm.rank == 0:
            print(f"  [adapt {it+1}/{niters}] celdas={nc:8d} | M={M:.10e} | "
                  f"err={err:.3e} | eta={eta_g:.3e} | Ieff={Ieff:.3f} | {wall:6.2f}s")
        if it < niters - 1:
            mesh = ex.refine_mesh(mesh, ex.dorfler_mark(eta_K, THETA_DORF))
    return rows, mesh


# ---------------------------------------------------------------------------
# Salida: CSV + grafica
# ---------------------------------------------------------------------------

def write_csv(unif, adap, M_ref):
    if comm.rank != 0:
        return
    fn = f"convergencia_{EXPERIMENT}.csv"
    with open(fn, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["modo", "ncells", "M_h", "err", "eta_dwr", "Ieff", "wall_s"])
        for r in unif:                       # excluye el ultimo (= referencia)
            err = abs(M_ref - r["M"])
            wr.writerow(["uniforme", r["ncells"], f'{r["M"]:.10e}',
                         f'{err:.6e}', "", "", f'{r["wall"]:.3f}'])
        for r in adap:
            wr.writerow(["adaptativo", r["ncells"], f'{r["M"]:.10e}',
                         f'{r["err"]:.6e}', f'{r["eta"]:.6e}',
                         f'{r["Ieff"]:.4f}', f'{r["wall"]:.3f}'])
    print(f"[csv] {fn}  (M_ref = {M_ref:.10e})")


def make_plot(unif, adap, M_ref):
    if comm.rank != 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib no disponible ({e}); usa el CSV.")
        return

    uc = np.array([r["ncells"] for r in unif], float)
    ue = np.array([abs(M_ref - r["M"]) for r in unif], float)
    ac = np.array([r["ncells"] for r in adap], float)
    ae = np.array([r["err"] for r in adap], float)
    um = ue > 0; am = ae > 0

    plt.figure(figsize=(6, 4.2))
    plt.loglog(uc[um], ue[um], "o-", color="C0", label="malla uniforme")
    plt.loglog(ac[am], ae[am], "s-", color="C1", label="malla adaptativa (DWR)")

    def _fit(xc, ye, color, name):
        if xc.size >= 2:
            p = np.polyfit(np.log(xc), np.log(ye), 1)
            xf = np.array([xc.min(), xc.max()])
            plt.loglog(xf, np.exp(p[1]) * xf**p[0], ":", color=color, lw=1.3,
                       label=fr"{name}: pend. {p[0]:.2f}")
    _fit(uc[um], ue[um], "C0", "ajuste uniforme")
    _fit(ac[am], ae[am], "C1", "ajuste adaptativa")

    if um.any():
        x0, y0 = uc[um][0], ue[um][0]
        xg = np.array([x0, x0 * 8.0])
        plt.loglog(xg, y0 * (xg / x0) ** (-1.0), "k--", lw=1,
                   label=r"$\mathcal{O}(h^2)$ (referencia)")

    plt.xlabel("numero de elementos")
    plt.ylabel(r"$|M(U)-M(U_h)|$")
    plt.title(f"Convergencia goal-oriented -- {EXPERIMENT}")
    plt.legend(); plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    fn = f"convergencia_{EXPERIMENT}.png"
    plt.savefig(fn, dpi=150)
    print(f"[plot] {fn}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if comm.rank == 0:
        print("=" * 70)
        print(f"  Convergencia uniforme vs adaptativo (mallas ANIDADAS) -- {EXPERIMENT}")
        print(f"  gauge=({ex.GAUGE_X0},{ex.GAUGE_Y0}) sigma={ex.GAUGE_SIGMA} | "
              f"dt={ex.DT} | Dorfler theta={THETA_DORF}")
        print("=" * 70)

    base_mesh = build_base()
    u_n0 = preroll_base(base_mesh)

    if comm.rank == 0:
        print("\n[1/2] secuencia uniforme anidada (el nivel mas fino = referencia)")
    unif_all, unif_meshes = run_uniform(u_n0, base_mesh, UNIFORM_LEVELS)
    M_ref = unif_all[-1]["M"]
    unif  = unif_all[:-1]                 # los graficados (sin la referencia)

    if comm.rank == 0:
        print("\n[2/2] ciclo adaptativo anidado")
    # build_base() de nuevo para arrancar el adaptativo desde una base limpia
    adap, mesh_adap = run_adaptive(u_n0, build_base(), M_ref, ADAPT_ITERS)

    # mallas para la figura
    lvl = min(SAVE_UNIF_LVL, len(unif_meshes) - 1)
    save_mesh(unif_meshes[lvl], f"malla_uniforme_{EXPERIMENT}.xdmf")
    save_mesh(mesh_adap,        f"malla_adaptativa_{EXPERIMENT}.xdmf")

    write_csv(unif, adap, M_ref)
    make_plot(unif, adap, M_ref)

    if comm.rank == 0:
        print("\n" + "=" * 70)
        print("  Hecho. convergencia_<exp>.csv/.png, "
              "malla_uniforme_<exp>.xdmf, malla_adaptativa_<exp>.xdmf")
        print("=" * 70)


if __name__ == "__main__":
    main()
