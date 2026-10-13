ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
import JuMP
import Ipopt
import MathOptInterface as MOI
import PythonCall
import MathOptAI as MOAI

# Requires torch and torch_geometric
PythonCall.pyimport("sys").path.append(pwd())
PythonCall.pyimport("vectorcanos")
predictor = MOAI.PytorchModel("vector-canos.pt")

N = 216
model = JuMP.Model(Ipopt.Optimizer)
#JuMP.set_optimizer_attribute(model, "linear_solver", "ma57")
JuMP.@variable(model, x[1:N], start = 0.5)
y, formulation = MOAI.add_predictor(model, predictor, x, gray_box = true)
xref = ones(N)
JuMP.@objective(model, Min, sum((x .- xref).^2))
# Arbitrary constraint on one CANOS output, just to exercise the embedded NN
JuMP.@constraint(model, y[2] >= 1.0)
JuMP.optimize!(model)
