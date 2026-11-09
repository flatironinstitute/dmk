using DMK
using BenchmarkTools

# module load gcc julia cuda/12.8
# julia
# ] dev /mnt/home/rblackwell/projects/codes/DMK/bindings/DMK.jl
#
# LD_LIBRARY_PATH=/path/to/dir/with/libdmk.so julia example.jl


n = 1000000;
r_src   = rand(Float32, 3n);
charge  = 2 .* rand(Float32, 3n) .- 1;
vel     = zeros(Float32, 3n);
params = PdmkParams()
params.n_dim    = 3;
params.kernel   = DMK_STOKESLET;
params.eval_src = DMK_VELOCITY;
params.eval_trg = DMK_VELOCITY;
params.eval_path = DMK_EVAL_PATH_GPU;
params.eps = 1e-5;

println("First time calling always takes a bit more time")
@time tree = pdmk_tree_createf(C_NULL, params, n, r_src, charge, nothing, 0, nothing)
println("Same with eval")
@time pdmk_tree_evalf(tree, vel, nothing)
pdmk_tree_destroy(tree)

println("Tree creation")
@time tree = pdmk_tree_createf(C_NULL, params, n, r_src, charge, nothing, 0, nothing)
println("Tree eval")
@time pdmk_tree_evalf(tree, vel, nothing)
charge .= 2 .* rand(3n) .- 1;
println("Charge update")
@time pdmk_tree_update_chargesf(tree, charge, nothing)
println("Tree eval")
@time pdmk_tree_evalf(tree, vel, nothing)
pdmk_tree_destroy(tree)

# --- ESP (Ewald Sum with PSWF kernels) ---
# Particles must lie in the unit box [0, 1)^n_dim.
r_esp   = rand(Float64, 3n);
charges = 2 .* rand(Float64, n) .- 1;
pot     = zeros(Float64, n);

esp = PdmkEspParams()
esp.n_dim  = 3;
esp.kernel = DMK_LAPLACE;
esp.eps    = 1e-6;

# Reuse a plan across many evaluations:
plan = pdmk_esp_plan_create(C_NULL, esp)
println("ESP eval")
@time pdmk_esp_eval(C_NULL, plan, n, r_esp, charges, nothing, pot)
pdmk_esp_plan_destroy(plan)

# Or the one-shot convenience wrapper (creates/destroys a plan internally):
pdmk_esp(C_NULL, esp, n, r_esp, charges, nothing, pot)

m = 5000;
r_ref  = rand(Float32, 3m);
q_ref  = 2 .* rand(Float32, m) .- 1;
pot_d  = zeros(Float32, m);
pot_t  = zeros(Float32, m);

dparams = PdmkParams()
dparams.n_dim     = 3;
dparams.kernel    = DMK_LAPLACE;
dparams.eval_src  = DMK_POTENTIAL;
dparams.eval_trg  = DMK_POTENTIAL;
dparams.eval_path = DMK_EVAL_PATH_GPU;
dparams.gpu_device_id = 0;
dparams.eps       = 1e-5;

println("Direct eval")
@time pdmk_directf(C_NULL, dparams, m, r_ref, q_ref, nothing, 0, nothing, pot_d, nothing)

pdmkf(C_NULL, dparams, m, r_ref, q_ref, nothing, 0, nothing, pot_t, nothing)

println("rel l2 error vs direct: ",
        sqrt(sum(abs2, pot_t .- pot_d) / sum(abs2, pot_d)))
