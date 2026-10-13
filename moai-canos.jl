ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
using JuMP
using Ipopt
using PythonCall
using Printf
import PowerModels
import PGLib
import HSL_jll

# Load file with model-building functions and utilities for collecting
# inputs and outputs
include("moai-canos-model.jl")

# Example: embed the trained CANOS-PF model in a PowerModels ACPF model at a
# training point, then compare CANOS outputs to ACPF outputs and data labels.
# Add your own objective/constraints on `inputs`, `outputs`, and `y` below.

# Python imports
PythonCall.pyimport("sys").path.append(pwd())
VC = PythonCall.pyimport("vectorcanos")
torch = PythonCall.pyimport("torch")

nn, predictor = load_canos("vectorcanos-trained.pt")

# Load dataset that we will use for target data
pfdelta_variants = PythonCall.pyimport("core.datasets.pfdelta_variants")
PFDeltaCANOS = pfdelta_variants.PFDeltaCANOS
dataset = PFDeltaCANOS(
    add_bus_type=true,
    case_name="case14",
    model="CANOS",
    root_dir=joinpath("data", "pfdelta_data"),
    split="train",
    task="1.1",
)

py_x0 = dataset[0]
py_x0_flat = nn.flatten_input(py_x0)
x0 = PythonCall.pyconvert(Vector{Float64}, py_x0_flat)
pglib_data = PGLib.pglib("case14")
load_pfd_into_pm!(pglib_data, py_x0)
pm = PowerModels.instantiate_model(pglib_data, PowerModels.ACPPowerModel, PowerModels.build_opf)

inputs, input_bounds = get_inputs(pm)
outputs, output_names, output_bounds = get_outputs(pm)
n_inputs = length(inputs)

# Keep only the ACPF equations, then re-add bounds on the inputs
strip_inequalities!(pm)
input_lbs = first.(input_bounds)
input_ubs = last.(input_bounds)
nonconst_mask = .!isa.(inputs, Number)
JuMP.@constraint(pm.model, input_lbs[nonconst_mask] .<= inputs[nonconst_mask] .<= input_ubs[nonconst_mask])

# Minimize 1-norm of difference between inputs and our target inputs.
JuMP.@variable(pm.model, input_slack_pos[1:n_inputs] >= 0.0, start = 0.0)
JuMP.@variable(pm.model, input_slack_neg[1:n_inputs] >= 0.0, start = 0.0)
JuMP.@constraint(pm.model, input_slack_eqn,
    inputs .- x0 .+ input_slack_pos .- input_slack_neg .== 0.0
)
JuMP.@objective(pm.model, Min, sum(input_slack_pos .+ input_slack_neg))

# CANOS outputs `y` are ordered like `outputs`
y, moai_inputs, _ = add_canos_predictor!(pm, predictor; start = x0)

JuMP.set_optimizer(pm.model, Ipopt.Optimizer)
JuMP.set_optimizer_attributes(pm.model, "linear_solver" => "ma27")
JuMP.optimize!(pm.model)
println("Termination status: $(JuMP.termination_status(pm.model))")

y_pf = JuMP.value.(outputs)
y_nn = JuMP.value.(y)
py_y_target = VC.flatten_input_labels(py_x0)
y_target = PythonCall.pyconvert(Vector{Float64}, py_y_target)

println()
println("Compare ACPF outputs, CANOS outputs, and data labels")
println("----------------------------------------------------")
println(@sprintf("%4s %14s %10s %10s %10s %10s", "idx", "name", "pf_out", "canos_out", "label", "nn_err"))
for i in eachindex(y_pf)
    println(@sprintf(
        "%4d %14s %10.4f %10.4f %10.4f %10.4f",
        i, output_names[i], y_pf[i], y_nn[i], y_target[i], y_nn[i] - y_pf[i],
    ))
end
