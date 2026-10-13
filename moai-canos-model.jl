ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
using JuMP
using Ipopt
using PythonCall
using PowerModels
using PGLib
import MathProgIncidence as MPIN
import MathOptAI as MOAI
import HSL_jll

PythonCall.pyimport("sys").path.append(@__DIR__)
VC = PythonCall.pyimport("vectorcanos")

"""
    load_pfd_into_pm!(pm_data::Dict, py_data::Py)

Populate a PowerModels data dict with PFDelta constants (loads and branch params)
from a PFDelta HeteroData sample.
"""
function load_pfd_into_pm!(pm_data::Dict, py_data::Py)
    # Set loads and slack bus params in pm_data
    load_matrix = PythonCall.pyconvert(Matrix{Float64}, py_data["bus"]["bus_demand"].numpy())
    gen_matrix = PythonCall.pyconvert(Matrix{Float64}, py_data["bus"]["bus_gen"].numpy())
    voltage_matrix = PythonCall.pyconvert(Matrix{Float64}, py_data["bus"]["bus_voltages"].numpy())
    slack_matrix = PythonCall.pyconvert(Matrix{Float64}, py_data["slack"]["x"].numpy())
    # We need to distribute these loads across all loads at each bus.
    loads_by_bus = Dict(b["index"] => Any[] for b in values(pm_data["bus"]))
    gens_by_bus = Dict(b["index"] => Any[] for b in values(pm_data["bus"]))
    nbus = length(pm_data["bus"])
    busindices = map(b -> b["index"], values(pm_data["bus"]))
    # Make sure bus indices are contiguous integers
    @assert all(sort(busindices) .== collect(1:nbus))
    for l in values(pm_data["load"])
        b = l["load_bus"]
        push!(loads_by_bus[b], l["index"])
    end
    for g in values(pm_data["gen"])
        b = g["gen_bus"]
        push!(gens_by_bus[b], g["index"])
    end
    for i in 1:length(pm_data["bus"])
        # Set vm and va for reference bus
        # Note that it doesn't matter that we have set any potential loads on the
        # reference bus. In fact, this will make it less confusing if we want to compare
        # net generation with actual generator values.
        if pm_data["bus"]["$i"]["bus_type"] == 3
            pm_data["bus"]["$i"]["va"] = slack_matrix[1, 1]
            pm_data["bus"]["$i"]["vm"] = slack_matrix[1, 2]
        end

        pd = load_matrix[i, 1]
        qd = load_matrix[i, 2]
        if pd != 0.0 || qd != 0.0
            nd = length(loads_by_bus[i])
            if nd == 0
                error("Zero loads attached to a bus that should have some net load")
            end
            p_per_load = pd / nd
            q_per_load = qd / nd
            for l in loads_by_bus[i]
                pm_data["load"]["$l"]["pd"] = p_per_load
                pm_data["load"]["$l"]["qd"] = q_per_load
            end
        end

        pg = gen_matrix[i, 1]
        vm = voltage_matrix[i, 2]
        if pg != 0.0
            # We are at a generator bus. pg and vm are degrees of freedom
            pm_data["bus"]["$i"]["vm"] = vm
            ng = length(gens_by_bus[i])
            if ng == 0 error("Zero generators at a bus with some net generation") end
            p_per_gen = pg / ng
            for g in gens_by_bus[i]
                pm_data["gen"]["$g"]["pg"] = p_per_gen
            end
        end
    end

    # Set line parameters
    branch_matrix = PythonCall.pyconvert(Matrix{Float64}, py_data["bus", "branch", "bus"]["edge_attr"])
    # This doesn't assume branch keys are contiguous integers
    branchkeys = sort(collect(keys(pm_data["branch"])); by=k -> parse(Int, k))
    for (i, k) in enumerate(branchkeys)
        br = pm_data["branch"][k]
        br["br_r"] = branch_matrix[i, 1]
        br["br_x"] = branch_matrix[i, 2]
        br["g_fr"] = branch_matrix[i, 3]
        br["b_fr"] = branch_matrix[i, 4]
        br["g_to"] = branch_matrix[i, 5]
        br["b_to"] = branch_matrix[i, 6]
        br["tap"] = branch_matrix[i, 7]
        br["shift"] = branch_matrix[i, 8]
    end

    return pm_data
end

"""
    get_outputs(pm::PowerModels.AbstractPowerModel)

Collect CANOS-style output variables/expressions from a PowerModels model, in
the same order as moai-canos.jl vectorization: bus (va, vm), PQ (va, vm),
PV (va, qg), slack (net_p, net_q), branch (pf, qf, pt, qt).

Returns a tuple `(outputs, names)` where `names` matches the ordering of
`outputs` (e.g., `va[1]`, `vm[1]`, `pf_fr[2]`, etc.).
"""
function get_outputs(pm::PowerModels.AbstractPowerModel)
    ref = pm.ref[:it][:pm][:nw][0]
    buskeys = sort(collect(keys(pm.data["bus"])); by=k -> parse(Int, k))
    branchkeys = sort(collect(keys(pm.data["branch"])); by=k -> parse(Int, k))

    bus_out = Any[]
    pq_out = Any[]
    pv_out = Any[]
    slack_out = Any[]
    branch_out = Any[]

    bus_names = String[]
    pq_names = String[]
    pv_names = String[]
    slack_names = String[]
    branch_names = String[]

    bus_bounds = Tuple{Float64,Float64}[]
    pq_bounds = Tuple{Float64,Float64}[]
    pv_bounds = Tuple{Float64,Float64}[]
    slack_bounds = Tuple{Float64,Float64}[]
    branch_bounds = Tuple{Float64,Float64}[]

    for buskey in buskeys
        idx = parse(Int, buskey)
        bus_type = pm.data["bus"][buskey]["bus_type"]

        va = PowerModels.var(pm, :va, idx)
        vm = PowerModels.var(pm, :vm, idx)
        append!(bus_out, [va, vm])
        append!(bus_names, ["va[$idx]", "vm[$idx]"])
        vmin = ref[:bus][idx]["vmin"]
        vmax = ref[:bus][idx]["vmax"]
        append!(bus_bounds, [(-2pi, 2pi), (vmin, vmax)])

        if bus_type == 1
            append!(pq_out, [va, vm])
            append!(pq_names, ["pq_va[$idx]", "pq_vm[$idx]"])
            append!(pq_bounds, [(-2pi, 2pi), (vmin, vmax)])
        elseif bus_type == 2
            qg = sum(PowerModels.var(pm, :qg, g) for g in ref[:bus_gens][idx]; init=0.0)
            qd = sum(ref[:load][l]["qd"] for l in ref[:bus_loads][idx]; init=0.0)
            net_q = qg - qd
            append!(pv_out, [net_q, va])
            append!(pv_names, ["pv_qg[$idx]", "pv_va[$idx]"])
            qmin = sum(ref[:gen][g]["qmin"] for g in ref[:bus_gens][idx]; init=0.0) - qd
            qmax = sum(ref[:gen][g]["qmax"] for g in ref[:bus_gens][idx]; init=0.0) - qd
            append!(pv_bounds, [(qmin, qmax), (-2pi, 2pi)])
        elseif bus_type == 3
            pg = sum(PowerModels.var(pm, :pg, g) for g in ref[:bus_gens][idx]; init=0.0)
            qg = sum(PowerModels.var(pm, :qg, g) for g in ref[:bus_gens][idx]; init=0.0)
            pd = sum(ref[:load][l]["pd"] for l in ref[:bus_loads][idx]; init=0.0)
            qd = sum(ref[:load][l]["qd"] for l in ref[:bus_loads][idx]; init=0.0)
            net_p = pg - pd
            net_q = qg - qd
            append!(slack_out, [net_p, net_q])
            append!(slack_names, ["slack_pg[$idx]", "slack_qg[$idx]"])
            pmin = sum(ref[:gen][g]["pmin"] for g in ref[:bus_gens][idx]; init=0.0) - pd
            pmax = sum(ref[:gen][g]["pmax"] for g in ref[:bus_gens][idx]; init=0.0) - pd
            qmin = sum(ref[:gen][g]["qmin"] for g in ref[:bus_gens][idx]; init=0.0) - qd
            qmax = sum(ref[:gen][g]["qmax"] for g in ref[:bus_gens][idx]; init=0.0) - qd
            append!(slack_bounds, [(pmin, pmax), (qmin, qmax)])
        else
            error("Unexpected bus type $(bus_type)")
        end
    end

    for branchkey in branchkeys
        idx = parse(Int, branchkey)
        fbus = ref[:branch][idx]["f_bus"]
        tbus = ref[:branch][idx]["t_bus"]
        append!(
            branch_out,
            [
                PowerModels.var(pm, :p, (idx, fbus, tbus)),
                PowerModels.var(pm, :q, (idx, fbus, tbus)),
                PowerModels.var(pm, :p, (idx, tbus, fbus)),
                PowerModels.var(pm, :q, (idx, tbus, fbus)),
            ],
        )
        append!(
            branch_names,
            ["pf_fr[$idx]", "qf_fr[$idx]", "pf_to[$idx]", "qf_to[$idx]"],
        )
        rate = get(ref[:branch][idx], "rate_a", Inf)
        flow_bounds = isfinite(rate) ? (-rate, rate) : (-Inf, Inf)
        append!(branch_bounds, [flow_bounds, flow_bounds, flow_bounds, flow_bounds])
    end

    outputs = vcat(bus_out, pq_out, pv_out, slack_out, branch_out)
    names = vcat(bus_names, pq_names, pv_names, slack_names, branch_names)
    bounds = vcat(bus_bounds, pq_bounds, pv_bounds, slack_bounds, branch_bounds)
    return outputs, names, bounds
end

function get_input_names(pm::PowerModels.AbstractPowerModel)
    buskeys = sort(collect(keys(pm.data["bus"])); by=k -> parse(Int, k))
    branchkeys = sort(collect(keys(pm.data["branch"])); by=k -> parse(Int, k))

    bus_names = String[]
    pq_names = String[]
    pv_names = String[]
    slack_names = String[]
    branch_names = String[]

    for i in buskeys
        idx = parse(Int, i)
        bus_type = pm.data["bus"][i]["bus_type"]
        if bus_type == 1
            append!(bus_names, ["bus_pd[$idx]", "bus_qd[$idx]"])
            append!(pq_names, ["pq_pd[$idx]", "pq_qd[$idx]"])
        elseif bus_type == 2
            append!(bus_names, ["bus_pg[$idx]", "bus_vm[$idx]"])
            append!(pv_names, ["pv_pg[$idx]", "pv_vm[$idx]"])
        elseif bus_type == 3
            append!(bus_names, ["bus_va[$idx]", "bus_vm[$idx]"])
            append!(slack_names, ["slack_va[$idx]", "slack_vm[$idx]"])
        else
            error("Unexpected bus type $(bus_type)")
        end
    end

    for i in branchkeys
        idx = parse(Int, i)
        append!(
            branch_names,
            [
                "br_r[$idx]",
                "br_x[$idx]",
                "g_fr[$idx]",
                "b_fr[$idx]",
                "g_to[$idx]",
                "b_to[$idx]",
                "tap[$idx]",
                "shift[$idx]",
            ],
        )
    end

    return vcat(bus_names, pq_names, pv_names, slack_names, branch_names)
end

"""
Get the vector of inputs as expected by CANOS-PF. These inputs can be
JuMP variables, JuMP expressions, or constants.
"""
function get_inputs(pm::PowerModels.AbstractPowerModel)
    # TODO: This function should also get the names of the inputs
    ref = pm.ref[:it][:pm][:nw][0]
    buskeys = sort(collect(keys(pm.data["bus"])); by=k -> parse(Int, k))
    branchkeys = sort(collect(keys(pm.data["branch"])); by=k -> parse(Int, k))
    # Inputs could be numbers, variables, or expressions
    bus_inputs = Any[]
    pq_inputs = Any[]
    pv_inputs = Any[]
    slack_inputs = Any[]
    branch_inputs = Any[]
    bus_bounds = Tuple{Float64,Float64}[]
    pq_bounds = Tuple{Float64,Float64}[]
    pv_bounds = Tuple{Float64,Float64}[]
    slack_bounds = Tuple{Float64,Float64}[]
    branch_bounds = Tuple{Float64,Float64}[]

    # Bus inputs
    # bus_type == 3 => reference bus
    # bus_type == 2 => generator (PV) bus
    # bus_type == 1 => load (PQ) bus
    for i in buskeys
        idx = parse(Int, i)
        bus_type = pm.data["bus"][i]["bus_type"]
        if bus_type == 1
            # Assume no generators live on a load bus
            @assert length(ref[:bus_gens][idx]) == 0
            pd = sum(ref[:load][l]["pd"] for l in ref[:bus_loads][idx]; init=0.0)
            qd = sum(ref[:load][l]["qd"] for l in ref[:bus_loads][idx]; init=0.0)
            append!(pq_inputs, [pd, qd])
            append!(bus_inputs, [pd, qd])
            # Add trivial bounds to avoid having to branch later on...
            append!(pq_bounds, [(pd, pd), (qd, qd)])
            append!(bus_bounds, [(pd, pd), (qd, qd)])
        elseif bus_type == 2
            # Since the input calls for total injection/demand, I sum up generator
            # active power variables at each node.
            # Note that PV buses can have loads as well.
            pg = (
                # We don't use init=0 here because there should always be a generator
                sum(PowerModels.var(pm, :pg, g) for g in ref[:bus_gens][idx])
                - sum(ref[:load][l]["pd"] for l in ref[:bus_loads][idx]; init = 0.0)
            )
            pgl = (
                sum(ref[:gen][i]["pmin"] for i in ref[:bus_gens][idx])
                - sum(ref[:load][l]["pd"] for l in ref[:bus_loads][idx]; init = 0.0)
            )
            pgu = (
                sum(ref[:gen][i]["pmax"] for i in ref[:bus_gens][idx])
                - sum(ref[:load][l]["pd"] for l in ref[:bus_loads][idx]; init = 0.0)
            )
            vm = PowerModels.var(pm, :vm, idx)
            vmin = ref[:bus][idx]["vmin"]
            vmax = ref[:bus][idx]["vmax"]
            append!(pv_inputs, [pg, vm])
            append!(bus_inputs, [pg, vm])
            append!(pv_bounds, [(pgl, pgu), (vmin, vmax)])
            append!(bus_bounds, [(pgl, pgu), (vmin, vmax)])
        elseif bus_type == 3
            va = PowerModels.var(pm, :va, idx)
            vm = PowerModels.var(pm, :vm, idx)
            vmin = ref[:bus][idx]["vmin"]
            vmax = ref[:bus][idx]["vmax"]
            append!(slack_inputs, [va, vm])
            append!(bus_inputs, [va, vm])
            append!(slack_bounds, [(-2pi, 2pi), (vmin, vmax)])
            append!(bus_bounds, [(-2pi, 2pi), (vmin, vmax)])
        else
            error("Unexpected bus type $(bus_type)")
        end
    end
    for i in branchkeys
        r = pm.data["branch"][i]["br_r"]
        x = pm.data["branch"][i]["br_x"]
        gf = pm.data["branch"][i]["g_fr"]
        bf = pm.data["branch"][i]["b_fr"]
        gt = pm.data["branch"][i]["g_to"]
        bt = pm.data["branch"][i]["b_to"]
        tap = pm.data["branch"][i]["tap"]
        shift = pm.data["branch"][i]["shift"]
        append!(branch_inputs, [r, x, gf, bf, gt, bt, tap, shift])
        append!(branch_bounds, [(r,r), (x,x), (gf,gf), (bf,bf), (gt,gt), (bt,bt), (tap,tap), (shift,shift)])
    end
    inputs = vcat(bus_inputs, pq_inputs, pv_inputs, slack_inputs, branch_inputs)
    bounds = vcat(bus_bounds, pq_bounds, pv_bounds, slack_bounds, branch_bounds)
    return inputs, bounds
end

"""
    strip_inequalities!(pm::PowerModels.AbstractPowerModel)

Delete all variable bounds and inequality constraints from `pm.model`, leaving
only the ACPF equality constraints. Re-add whatever bounds you need afterwards
(e.g., from the bounds returned by `get_inputs`/`get_outputs`).
"""
function strip_inequalities!(pm::PowerModels.AbstractPowerModel)
    for var in JuMP.all_variables(pm.model)
        if JuMP.has_lower_bound(var)
            JuMP.delete_lower_bound(var)
        end
        if JuMP.has_upper_bound(var)
            JuMP.delete_upper_bound(var)
        end
    end
    for con in MPIN.get_inequality_constraints(pm.model)
        JuMP.delete(pm.model, con)
    end
    return pm
end

"""
    solve_powerflow(point)::Vector{Float64}

Solve AC power flow for a PFDelta point and return a vector of outputs ordered
like the vectorized CANOS outputs.
"""
function solve_powerflow(point; silent = true)
    # Build a PowerModels case from PGLib and overwrite constants from PFDelta point
    pm_data = PGLib.pglib("case14")  # assumes case14; adjust if other cases are used
    load_pfd_into_pm!(pm_data, point)

    # For some reason, solving OPF with min-error-to-x0 gives the correct result
    # (i.e., matches the labels from the input data, `point`), while solving PF
    # gives an incorrect result. Likely I'm setting some degree of freedom improperly?

    # build-opf implementation
    # ------------------------
    pm = PowerModels.instantiate_model(pm_data, PowerModels.ACPPowerModel, PowerModels.build_opf)
    strip_inequalities!(pm)
    inputs, input_bounds = get_inputs(pm)
    n_inputs = length(inputs)
    outputs, output_names, output_bounds = get_outputs(pm)
    n_outputs = length(outputs)

    py_x0 = VC.flatten_input(point).numpy()
    x0 = PythonCall.pyconvert(Vector{Float64}, py_x0)
    # Minimize 1-norm of difference between inputs and our target inputs.
    JuMP.@variable(pm.model, input_slack_pos[1:n_inputs] >= 0.0, start = 0.0)
    JuMP.@variable(pm.model, input_slack_neg[1:n_inputs] >= 0.0, start = 0.0)
    JuMP.@constraint(pm.model, input_slack_eqn,
        inputs .- x0 .+ input_slack_pos .- input_slack_neg .== 0.0
    )
    JuMP.@objective(pm.model, Min, sum(input_slack_pos .+ input_slack_neg))

    # build-pf implementation
    # -----------------------
    #pm = PowerModels.instantiate_model(pm_data, PowerModels.ACPPowerModel, PowerModels.build_pf)
    #outputs, output_names = get_outputs(pm)

    ipopt = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "linear_solver" => "ma27")
    JuMP.set_optimizer(pm.model, ipopt)
    if silent
        JuMP.set_silent(pm.model)
    end
    JuMP.optimize!(pm.model)
    return JuMP.value.(outputs)
end

"""
    load_canos(modelpath = "vectorcanos-trained.pt")

Load a saved `VectorCanos` wrapper. Returns `(nn, predictor)`, where `nn` is the
Python module (for direct forward passes, `flatten_input`, etc.) and `predictor`
is a `MathOptAI.PytorchModel` for embedding in a JuMP model.
"""
function load_canos(modelpath::AbstractString = "vectorcanos-trained.pt")
    torch = PythonCall.pyimport("torch")
    nn = torch.load(modelpath, map_location="cpu")
    predictor = MOAI.PytorchModel(modelpath)
    return nn, predictor
end

"""
    add_canos_predictor!(pm, predictor; start = 1.0, device = nothing)

Embed a vectorized CANOS model in `pm.model`. The NN inputs are linked to the
PowerModels quantities returned by `get_inputs(pm)`, so the returned outputs `y`
are ordered like `get_outputs(pm)`. `start` is a scalar or a vector of initial
values for the NN inputs. `device` defaults to "cuda" if available, else "cpu".

Returns `(y, moai_inputs, formulation)`.
"""
function add_canos_predictor!(
    pm::PowerModels.AbstractPowerModel,
    predictor;
    start = 1.0,
    device = nothing,
)
    inputs, _ = get_inputs(pm)
    n_inputs = length(inputs)
    # Inputs may be constants or expressions, but MathOptAI needs variables, so we
    # link them to auxiliary variables.
    moai_inputs = JuMP.@variable(
        pm.model,
        [i = 1:n_inputs],
        start = start isa Number ? start : start[i],
        base_name = "moai_inputs",
    )
    JuMP.@constraint(pm.model, inputs .== moai_inputs)
    if isnothing(device)
        torch = PythonCall.pyimport("torch")
        device = PythonCall.pyconvert(Bool, torch.cuda.is_available()) ? "cuda" : "cpu"
    end
    y, formulation = MOAI.add_predictor(pm.model, predictor, moai_inputs; gray_box = true, device)
    return y, moai_inputs, formulation
end
