using LinearAlgebra, HDF5, PythonCall

# File path
h5_path = "./pickles/posterior_full_synth_NA_tmean_sm_2km.pkl.h5"
out_path = "./pickles/L_full_synth_NA_tmean_sm_2km.h5"

# --- Load and symmetrize ---
C = h5open(h5_path, "r") do f
    Array{Float64}(read(f["C"]))   # dataset name is "C"
end
C = 0.5 .* (C .+ C')
println("Loaded C: ", size(C))

# --- LDLᵀ with rook pivoting ---
bk = bunchkaufman(Symmetric(C, :L), true)
println("LDLᵀ factorization complete.")

# Choleky factorization (check positive definite) it isnt for 1km
#L = cholesky(Symmetric(C, :L) + 1e-8I).L
L = bk.L

println("L size: ", size(L))

h5open(out_path, "w") do f
    write(f, "L", Matrix(L))      # dense lower-triangular (unit diagonal)
end
println("Saved L to $out_path")