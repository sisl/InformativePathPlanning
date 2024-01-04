##########################################################################
# d-optimal
##########################################################################
function run_d_optimal_sensor_selection(mmipp::MultimodalIPP, path::Vector{Int})
    ipp_problem = mmipp.ipp_problem
    n = ipp_problem.n
    m = ipp_problem.m
    G = ipp_problem.Graph.G
    Theta = ipp_problem.Graph.Theta
    Omega = ipp_problem.Graph.Omega
    Σx = ipp_problem.MeasurementModel.Σₓ
    Σₓ⁻¹ = ipp_problem.MeasurementModel.Σₓ⁻¹
    A = ipp_problem.MeasurementModel.A
    σ_min = mmipp.σ_min
    σ_max = mmipp.σ_max
    B = ipp_problem.B
    start = ipp_problem.Graph.start
    end_idx = ipp_problem.Graph.goal
    dist = ipp_problem.Graph.distances
    k_sensors = mmipp.k
    timeout = ipp_problem.solution_time
    
    model = Model(Mosek.Optimizer)

    @variable(model, 0 <= s[1:n] <= 1)
    @variable(model, t)

    # Objective
    @objective(model, Max, t)

    ZA_mat = σ_max^(-2)*sum((i in path)*A[i, :]*A[i, :]' for i in 1:n)
    Σₚ⁻¹ = ZA_mat + Σₓ⁻¹
    Σ_map_inv = JuMP.@expression(model, (σ_min^(-1))*sum(s[i]*A[i, :]*A[i, :]' for i in 1:n) + Σₚ⁻¹) # note we don't square σ_min for numerical stability since it is already quite small

    @constraint(model, [t, 1, (Σ_map_inv[i, j] for i in 1:m for j in 1:i)...] in MOI.LogDetConeTriangle(m))

    idx_not_on_path = [i for i in 1:n if i ∉ path]
    @constraint(model, [i in idx_not_on_path], s[i] == 0)
    @constraint(model, sum(s[i] for i in 1:n) == k_sensors)

    set_time_limit_sec(model, timeout)
    optimize!(model)

    if termination_status(model) != OPTIMAL
        @warn "Model not solved to optimality. Terminated with status: $(termination_status(model))"
    end

    # take top k_sensor values from z
    optimal_s = JuMP.value.(model[:s])
    indices_values = [(i, optimal_s[i]) for i in path]
    sorted_indices_values = sort(indices_values, by=x->x[2], rev=true)[1:min(k_sensors, length(indices_values))]
    drills = [tuple[1] for tuple in sorted_indices_values]

    @show sum(optimal_s[i] for i in path)

    return drills
end

##########################################################################
# a-optimal
##########################################################################
function is_positive_semidefinite(A::AbstractMatrix)
    eigvals = LinearAlgebra.eigvals(A)
    return all(λ -> λ ≥ 0, eigvals)
end

function run_a_optimal_sensor_selection(mmipp::MultimodalIPP, path::Vector{Int})
    ipp_problem = mmipp.ipp_problem
    n = ipp_problem.n
    m = ipp_problem.m
    G = ipp_problem.Graph.G
    Theta = ipp_problem.Graph.Theta
    Omega = ipp_problem.Graph.Omega
    Σx = ipp_problem.MeasurementModel.Σₓ
    Σₓ⁻¹ = ipp_problem.MeasurementModel.Σₓ⁻¹
    A = ipp_problem.MeasurementModel.A
    σ_min = mmipp.σ_min
    σ_max = mmipp.σ_max
    B = ipp_problem.B
    start = ipp_problem.Graph.start
    end_idx = ipp_problem.Graph.goal
    dist = ipp_problem.Graph.distances
    k_sensors = mmipp.k
    timeout = ipp_problem.solution_time
    
    model = Model(Mosek.Optimizer)

    @variable(model, 0 <= s[1:n] <= 1)

    @variable(model, Y[1:m, 1:m], PSD)

    # Objective
    @objective(model, Min, tr(Y))

    ZA_mat = σ_max^(-2)*sum((i in path)*A[i, :]*A[i, :]' for i in 1:n)
    Σₚ⁻¹ = ZA_mat + Σₓ⁻¹
    Σ_map_inv = JuMP.@expression(model, (σ_min^(-1))*sum(s[i]*A[i, :]*A[i, :]' for i in 1:n) + Σₚ⁻¹) # note we don't square σ_min for numerical stability since it is already quite small
    
    @constraint(model, [Y Diagonal(ones(m)); Diagonal(ones(m)) Σ_map_inv] >= 0, PSDCone())

    idx_not_on_path = [i for i in 1:n if i ∉ path]
    @constraint(model, [i in idx_not_on_path], s[i] == 0)
    @constraint(model, sum(s[i] for i in 1:n) == k_sensors)

    set_time_limit_sec(model, timeout)
    optimize!(model)

    if termination_status(model) != OPTIMAL
        @warn "Model not solved to optimality. Terminated with status: $(termination_status(model))"
    end
    if !is_positive_semidefinite(JuMP.value.(Σ_map_inv))
        @warn "Σ_map_inv PSDCone() constraint NOT satisfied. Numerical issues likely encountered."
    end

    # take top k_sensor values from S
    optimal_s = JuMP.value.(model[:s])
    indices_values = [(i, optimal_s[i]) for i in path]
    sorted_indices_values = sort(indices_values, by=x->x[2], rev=true)[1:min(k_sensors, length(indices_values))]
    drills = [tuple[1] for tuple in sorted_indices_values]

    @show sum(optimal_s[i] for i in path)

    return drills
end