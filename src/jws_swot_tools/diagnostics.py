import numpy as np
import matplotlib.pyplot as plt

# checks if aa matrix is positive definite and plots eigenvalues 
def diagnose_positive_definite(R_tt, ny=None, nx=None, n_modes=4):
    print("=== Positive Definite Diagnostic ===\n")

    # 1) Symmetry check
    is_symmetric = np.allclose(R_tt, R_tt.T)
    print(f"Symmetric: {is_symmetric}")
    if not is_symmetric:
        print("  ISSUE: Matrix must be symmetric for positive definiteness")
        return

    # 2) Eigenvalues / eigenvectors
    print("\n--- Eigenvalue Analysis ---")
    eigenvalues, eigenvectors = np.linalg.eigh(R_tt)
    eig_min, eig_max = float(eigenvalues[0]), float(eigenvalues[-1])
    print(f"Smallest eigenvalue: {eig_min:.6e}")
    print(f"Largest eigenvalue:  {eig_max:.6e}")

    neg_eigs = eigenvalues[eigenvalues < 0]
    if neg_eigs.size > 0:
        print(f"\n   ISSUE: {neg_eigs.size} negative eigenvalues")
        print(f"     Range: [{neg_eigs.min():.3e}, {neg_eigs.max():.3e}]")

    cond = eig_max / max(abs(eig_min), 1e-300)
    print(f"\nCondition number: {cond:.2e}")
    if cond > 1e10:
        print(" Poorly conditioned (nearly singular)")

    if eig_min > 1e-10:
        print("POSITIVE DEFINITE")
    elif eig_min > -1e-10:
        print("POSITIVE SEMIDEFINITE")
    else:
        print("INDEFINITE (negative eigenvalues)")

    # 3) eigenvalue spectrum 
    idx = np.arange(eigenvalues.size)
    pos = eigenvalues > 0
    neg = eigenvalues < 0
    zer = ~pos & ~neg

    plt.figure(figsize=(7,4))
    plt.scatter(idx[pos], eigenvalues[pos], s=6, label="λ > 0")
    plt.scatter(idx[neg], eigenvalues[neg], s=6, c="red", label="λ < 0")
    if zer.any():
        plt.scatter(idx[zer], np.zeros(zer.sum()), s=6, c="k", label="≈0")

    plt.axhline(0, color="k", lw=0.7)
    plt.yscale("symlog", linthresh=1e-12)  # shows ± values; linear near 0
    plt.xlabel("Index"); plt.ylabel("λ")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("eigenvalues_symlog.png", dpi=150)

    return eigenvalues, eigenvectors