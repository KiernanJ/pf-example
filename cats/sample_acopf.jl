#=
Generate AC OPF samples for the California Test System (CATS) in PFΔ's raw
sample format, for training CANOS-PF.

Sampling procedure
------------------
Sample i (1-based) draws from its own RNG seeded from (--seed, i), so runs are
reproducible and can be split across cluster jobs with --start without overlap.

1. Loads. Each load is scaled by a system-wide factor α ~ U(--alpha-min, --alpha-max)
   (correlated variation, e.g., time of day) times independent per-load factors
   ηp, ηq ~ LogNormal(0, --sigma) on P and Q separately (local variation, which
   also varies each load's power factor):
       pd = α ηp pd0,   qd = α ηq qd0
2. Costs (--costs shuffle). Generator cost functions are randomly permuted, as
   in PFΔ, so that the dispatch varies and not just the loads.
3. Solve AC OPF with Ipopt, warm-started from the base-case solution.
4. Accept only LOCALLY_SOLVED/OPTIMAL solves (plus ALMOST_* with --accept-almost)
   whose power balance, recomputed from the solved voltages, has a max
   mismatch below --pb-tol (p.u.). Rejected samples are recorded rather than
   silently dropped, so the acceptance rate can be checked and reported.

Formulations (--formulation):
  pfdelta  (default) PFΔ's relaxed formulation from
           data_generation/src/build_opf_pfdelta.jl: voltage bounds only at PV/slack
           buses, qg unbounded, no thermal limits. This is how PFΔ's CANOS training
           data was generated. On CATS it accepted 4/4 test samples at ~9 s each.
  acopf    Standard PowerModels AC OPF with all operating limits. On CATS with
           MUMPS it accepted 1/4 of the same samples (LOCALLY_INFEASIBLE or
           TIME_LIMIT otherwise) at ~67 s each.

Outputs (under --out):
  n/raw/sample_<i>.json       Accepted samples: {"network", "solution", "parent_seed",
                              "sample_meta"}, as read by PFDeltaDataset.build_heterodata.
  n/rejected/sample_<i>.json  Metadata of rejected samples.
  base_case.json              Base-case network and solution.
  summary_<first>-<last>.csv  One row per attempted sample.
  config_<first>-<last>.json  Options and package versions used.

Files are written atomically and re-running skips samples that already have an
output file, so an interrupted job can be restarted with the same arguments.

Usage
-----
  julia --project=cats -e 'using Pkg; Pkg.instantiate()'
  julia --project=cats cats/sample_acopf.jl --num-samples 1000 --workers 12

For SLURM array jobs, give each task its own --start, e.g.
  --start $((SLURM_ARRAY_TASK_ID * 1000 + 1)) --num-samples 1000 --workers $SLURM_CPUS_PER_TASK

HSL linear solvers (--linear-solver ma27/ma57/ma86/ma97) require a licensed
HSL_jll added to this environment; the default is MUMPS.

Options (defaults in DEFAULTS below):
  --case PATH            MATPOWER case file
  --out DIR              Output directory (default data/pfdelta_data/cats)
  --num-samples N        Number of samples to attempt
  --start I              Index of the first sample
  --workers N            Number of parallel worker processes
  --seed S               Base random seed
  --alpha-min A          Lower bound of the system-wide load factor
  --alpha-max A          Upper bound of the system-wide load factor
  --sigma S              Std. dev. of log per-load factors
  --costs shuffle|none   Generator cost perturbation
  --formulation pfdelta|acopf
  --linear-solver NAME   Ipopt linear solver
  --tol T                Ipopt tolerance
  --max-cpu-time T       Ipopt time limit per solve, in seconds
  --pb-tol T             Max accepted power-balance mismatch, in p.u.
  --accept-almost        Also accept ALMOST_LOCALLY_SOLVED/ALMOST_OPTIMAL
  --no-warm-start        Solve each sample from PowerModels' default start
=#

using Distributed

const DEFAULTS = Dict{String,Any}(
    "case" => joinpath(@__DIR__, "CaliforniaTestSystem.m"),
    "out" => joinpath("data", "pfdelta_data", "cats"),
    "num-samples" => 100,
    "start" => 1,
    "workers" => 1,
    "seed" => 1234,
    "alpha-min" => 0.8,
    "alpha-max" => 1.2,
    "sigma" => 0.05,
    "costs" => "shuffle",
    "formulation" => "pfdelta",
    "linear-solver" => "mumps",
    "tol" => 1e-6,
    "max-cpu-time" => 600.0,
    "pb-tol" => 1e-4,
    "accept-almost" => false,
    "warm-start" => true,
)

function parse_args(args)
    opts = copy(DEFAULTS)
    flags = Dict("--accept-almost" => ("accept-almost", true), "--no-warm-start" => ("warm-start", false))
    i = 1
    while i <= length(args)
        arg = args[i]
        if haskey(flags, arg)
            key, value = flags[arg]
            opts[key] = value
            i += 1
            continue
        end
        key = startswith(arg, "--") ? arg[3:end] : ""
        if !haskey(DEFAULTS, key) || DEFAULTS[key] isa Bool || i == length(args)
            error("Unknown or incomplete option: $arg")
        end
        default = DEFAULTS[key]
        opts[key] = default isa AbstractString ? args[i+1] : parse(typeof(default), args[i+1])
        i += 2
    end
    opts["costs"] in ("shuffle", "none") || error("--costs must be shuffle or none")
    opts["formulation"] in ("pfdelta", "acopf") || error("--formulation must be pfdelta or acopf")
    opts["alpha-min"] <= opts["alpha-max"] || error("--alpha-min must be <= --alpha-max")
    return opts
end

const OPTS = parse_args(ARGS)

if OPTS["workers"] > 1
    # One BLAS/MUMPS thread per worker; parallelism comes from the workers
    ENV["OMP_NUM_THREADS"] = "1"
    addprocs(OPTS["workers"]; exeflags = "--project=$(Base.active_project())")
end

@everywhere begin
    using PowerModels
    using Ipopt
    using JuMP
    using JSON
    using Random
    using Printf
    import LinearAlgebra
    const PM = PowerModels
    PM.silence()
    LinearAlgebra.BLAS.set_num_threads(1)
end

if startswith(OPTS["linear-solver"], "ma")
    @everywhere import HSL_jll
end
if OPTS["formulation"] == "pfdelta"
    const PFDELTA_BUILDER = joinpath(@__DIR__, "..", "data_generation", "src", "build_opf_pfdelta.jl")
    @everywhere begin
        using PowerModels: ids, ref, var
        include($PFDELTA_BUILDER)
    end
end

@everywhere begin

const SOLVED = [JuMP.LOCALLY_SOLVED, JuMP.OPTIMAL]
const ALMOST_SOLVED = [JuMP.ALMOST_LOCALLY_SOLVED, JuMP.ALMOST_OPTIMAL]
const START_KEYS = ("vm_start", "va_start", "pg_start", "qg_start")

sorted_keys(d) = sort(collect(keys(d)); by = k -> parse(Int, k))

function make_optimizer(opts)
    attrs = Pair{String,Any}[
        "linear_solver" => opts["linear-solver"],
        "tol" => opts["tol"],
        "max_cpu_time" => opts["max-cpu-time"],
        "print_level" => 0,
    ]
    if startswith(opts["linear-solver"], "ma")
        push!(attrs, "hsllib" => HSL_jll.libhsl_path)
    end
    return JuMP.optimizer_with_attributes(Ipopt.Optimizer, attrs...)
end

function solve_opf(net, opts)
    build = opts["formulation"] == "pfdelta" ? build_opf_power_flow_delta : PM.build_opf
    return PM.solve_model(net, PM.ACPPowerModel, make_optimizer(opts), build)
end

"Scale loads by a system-wide factor times per-load lognormal factors. Returns α."
function perturb_loads!(net, base, rng, opts)
    α = opts["alpha-min"] + (opts["alpha-max"] - opts["alpha-min"]) * rand(rng)
    σ = opts["sigma"]
    for k in sorted_keys(base["load"])
        net["load"][k]["pd"] = α * exp(σ * randn(rng)) * base["load"][k]["pd"]
        net["load"][k]["qd"] = α * exp(σ * randn(rng)) * base["load"][k]["qd"]
    end
    return α
end

"Randomly permute generator cost functions (as in PFΔ)."
function shuffle_costs!(net, base, rng)
    keys_ = sorted_keys(base["gen"])
    perm = randperm(rng, length(keys_))
    for (k, j) in zip(keys_, keys_[perm])
        for field in ("model", "ncost", "cost")
            net["gen"][k][field] = deepcopy(base["gen"][j][field])
        end
    end
    return
end

function set_warm_start!(net, solution)
    for (k, bus) in solution["bus"]
        net["bus"][k]["vm_start"] = bus["vm"]
        net["bus"][k]["va_start"] = bus["va"]
    end
    for (k, gen) in solution["gen"]
        net["gen"][k]["pg_start"] = gen["pg"]
        net["gen"][k]["qg_start"] = gen["qg"]
    end
    return
end

function strip_warm_start!(net)
    for comp in ("bus", "gen"), item in values(net[comp]), key in START_KEYS
        delete!(item, key)
    end
    return
end

"Max power-balance mismatch (p.u.) with branch flows recomputed from the solved voltages."
function max_mismatch(net, solution)
    data = deepcopy(net)
    PM.update_data!(data, solution)
    PM.update_data!(data, PM.calc_branch_flow_ac(data))
    balance = PM.calc_power_balance(data)
    return maximum(max(abs(b["p_delta"]), abs(b["q_delta"])) for b in values(balance["bus"]))
end

"Make the PowerModels result JSON-safe (status enums to strings, non-finite numbers to null)."
function json_result(result)
    out = Dict{String,Any}()
    for (k, v) in result
        if v isa Enum
            out[k] = string(v)
        elseif v isa Real && !isfinite(v)
            out[k] = nothing
        else
            out[k] = v
        end
    end
    return out
end

function write_atomic(path, obj)
    mkpath(dirname(path))
    tmp = path * ".tmp"
    open(io -> JSON.print(io, obj), tmp, "w")
    mv(tmp, path; force = true)
    return
end

function sample_paths(out, i)
    return joinpath(out, "n", "raw", "sample_$i.json"), joinpath(out, "n", "rejected", "sample_$i.json")
end

function run_sample(i, base, base_solution, opts)
    raw_path, rejected_path = sample_paths(opts["out"], i)
    if isfile(raw_path) || isfile(rejected_path)
        return (; index = i, status = "skipped", accepted = isfile(raw_path), alpha = NaN,
                objective = NaN, solve_time = NaN, max_mismatch = NaN, wall_time = 0.0)
    end
    t0 = time()
    seed = opts["seed"] * 1_000_000_007 + i
    rng = Random.Xoshiro(seed)
    net = deepcopy(base)
    alpha = perturb_loads!(net, base, rng, opts)
    opts["costs"] == "shuffle" && shuffle_costs!(net, base, rng)
    opts["warm-start"] && set_warm_start!(net, base_solution)

    status, objective, solve_time, mismatch = "error", NaN, NaN, NaN
    result = nothing
    try
        result = solve_opf(net, opts)
        status = string(result["termination_status"])
        objective = result["objective"]
        solve_time = result["solve_time"]
        ok = result["termination_status"] in SOLVED ||
             (opts["accept-almost"] && result["termination_status"] in ALMOST_SOLVED)
        if ok
            mismatch = max_mismatch(net, result["solution"])
            mismatch <= opts["pb-tol"] || (status = "power_balance_mismatch")
        end
    catch err
        status = "error: " * sprint(showerror, err)
    end
    accepted = status in string.(SOLVED) ||
               (opts["accept-almost"] && status in string.(ALMOST_SOLVED))
    accepted = accepted && mismatch <= opts["pb-tol"]

    meta = Dict("index" => i, "seed" => seed, "alpha" => alpha, "status" => status,
                "objective" => isfinite(objective) ? objective : nothing,
                "solve_time" => isfinite(solve_time) ? solve_time : nothing,
                "max_mismatch" => isfinite(mismatch) ? mismatch : nothing,
                "formulation" => opts["formulation"], "costs" => opts["costs"])
    if accepted
        strip_warm_start!(net)
        write_atomic(raw_path, Dict("network" => net, "solution" => json_result(result),
                                    "parent_seed" => -1, "sample_meta" => meta))
    else
        write_atomic(rejected_path, Dict("sample_meta" => meta))
    end
    wall_time = time() - t0
    println(@sprintf("sample %d: %s, alpha=%.3f, solve=%.1fs, mismatch=%.1e (worker %d)",
                     i, accepted ? "accepted" : "rejected [$status]", alpha, solve_time, mismatch, myid()))
    flush(stdout)
    return (; index = i, status, accepted, alpha, objective, solve_time, max_mismatch = mismatch, wall_time)
end

end # @everywhere

function main(opts)
    first_i = opts["start"]
    last_i = first_i + opts["num-samples"] - 1
    tag = "$(first_i)-$(last_i)"
    mkpath(opts["out"])
    println("CATS AC OPF sampling: samples $tag, $(nworkers()) worker(s), output in $(opts["out"])")

    versions = Dict("julia" => string(VERSION), "PowerModels" => string(pkgversion(PowerModels)),
                    "Ipopt" => string(pkgversion(Ipopt)), "JuMP" => string(pkgversion(JuMP)))
    write_atomic(joinpath(opts["out"], "config_$tag.json"), Dict("options" => opts, "versions" => versions))

    # Base case: preflight check and warm start for every sample
    base = PM.parse_file(opts["case"])
    t = @elapsed base_result = solve_opf(base, opts)
    base_status = base_result["termination_status"]
    println("Base case: $base_status, objective=$(base_result["objective"]), $(round(t, digits = 1)) s")
    base_status in SOLVED || error("Base case did not solve ($base_status); check the solver setup")
    write_atomic(joinpath(opts["out"], "base_case.json"),
                 Dict("network" => base, "solution" => json_result(base_result)))
    base_solution = base_result["solution"]

    indices = collect(first_i:last_i)
    rows = pmap(i -> run_sample(i, base, base_solution, opts), CachingPool(workers()), indices)

    summary_path = joinpath(opts["out"], "summary_$tag.csv")
    open(summary_path, "w") do io
        println(io, "index,status,accepted,alpha,objective,solve_time,max_mismatch,wall_time")
        for r in rows
            println(io, join((r.index, "\"$(r.status)\"", r.accepted, r.alpha, r.objective,
                              r.solve_time, r.max_mismatch, r.wall_time), ","))
        end
    end

    attempted = filter(r -> r.status != "skipped", rows)
    n_ok = count(r -> r.accepted, attempted)
    println()
    println("Attempted $(length(attempted)) samples ($(length(rows) - length(attempted)) skipped as already done)")
    if !isempty(attempted)
        println(@sprintf("Accepted %d (%.1f%%)", n_ok, 100 * n_ok / length(attempted)))
        times = [r.solve_time for r in attempted if isfinite(r.solve_time)]
        isempty(times) || println(@sprintf("Mean solve time %.1f s", sum(times) / length(times)))
    end
    println("Wrote $summary_path")
end

main(OPTS)
