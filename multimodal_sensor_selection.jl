##########################################################################
# d-optimal
##########################################################################
function run_d_optimal_sensor_selection(mipp::MultimodalIPP, path::Vector{Int})
    ipp_problem = mipp.ipp_problem
    N = ipp_problem.n
    M = ipp_problem.m
    G = ipp_problem.Graph.G
    Theta = ipp_problem.Graph.Theta
    Omega = ipp_problem.Graph.Omega
    Σx = ipp_problem.MeasurementModel.Σₓ
    Σx_inv = ipp_problem.MeasurementModel.Σₓ⁻¹
    σ_min = mipp.σ_min
    σ_max = mipp.σ_max
    B = ipp_problem.B
    start = ipp_problem.Graph.start
    end_idx = ipp_problem.Graph.goal
    dist = ipp_problem.Graph.distances
    k_sensors = mipp.k
    timeout = ipp_problem.solution_time
    
    model = Model(Mosek.Optimizer)

    @variable(model, 0 <= s[1:N] <= 1)
    @variable(model, t)

    # Objective
    @objective(model, Max, t)

    ZA_mat = σ^(-2)*sum((i in path)*A[i, :]*A[i, :]' for i in 1:n)
    Σₚ⁻¹ = ZA_mat + Σₓ⁻¹
    
    Σ_map_inv = JuMP.@expression(model, (Diagonal(s[pred_idx] .* 1/σ_min) + Σₚ⁻¹)) # note we don't square σ_min for numerical stability since it is already quite small



###############################################################


    
    # this uses the fact that A[:,i]*A[:,i]' is just I since A itself is I
    zaaᵀ = zeros(N)
    zaaᵀ[path] .= 1.0
    
    # this is our posterior from the path which becomes our prior for sample selection
    Σₚ⁻¹ = Σx_inv + Diagonal(σ_max^(-2) .* zaaᵀ)

    pred_idx = find_closest_points(Theta, Omega)

    # Σ_map_inv = JuMP.@expression(model, (Diagonal(z .* 1/σ_min) + Σₚ⁻¹)[pred_idx, pred_idx])
    Σ_map_inv = JuMP.@expression(model, (Diagonal(z[pred_idx] .* 1/σ_min) + Σₚ⁻¹[pred_idx, pred_idx])) # note we don't square σ_min for numerical stability since it is already quite small

    @constraint(model, [t, 1, (Σ_map_inv[i, j] for i in 1:M for j in 1:i)...] in MOI.LogDetConeTriangle(M))

    idx_not_on_path = [i for i in 1:N if i ∉ path]
    @constraint(model, [i in idx_not_on_path], z[i] == 0)
    @constraint(model, sum(z[i] for i in 1:N) == k_sensors)

    set_time_limit_sec(model, timeout)
    optimize!(model)

    if termination_status(model) != OPTIMAL
        @warn "Model not solved to optimality. Terminated with status: $(termination_status(model))"
    end

    # take top k_sensor values from z
    optimal_z = JuMP.value.(model[:z])
    indices_values = [(i, optimal_z[i]) for i in path]
    sorted_indices_values = sort(indices_values, by=x->x[2], rev=true)[1:min(k_sensors, length(indices_values))]
    drills = [tuple[1] for tuple in sorted_indices_values]

    @show sum(optimal_z[i] for i in path)

    return drills
end

##########################################################################
# a-optimal
##########################################################################
function is_positive_semidefinite(A::AbstractMatrix)
    eigvals = LinearAlgebra.eigvals(A)
    return all(λ -> λ ≥ 0, eigvals)
end

function run_a_optimal_sensor_selection(mipp::MultimodalIPP, path::Vector{Int})
    ipp_problem = mipp.ipp_problem
    N = ipp_problem.n
    M = ipp_problem.m
    G = ipp_problem.Graph.G
    Theta = ipp_problem.Graph.Theta
    Omega = ipp_problem.Graph.Omega
    Σx = ipp_problem.MeasurementModel.Σₓ
    Σx_inv = ipp_problem.MeasurementModel.Σₓ⁻¹
    σ_min = mipp.σ_min
    σ_max = mipp.σ_max
    B = ipp_problem.B
    start = ipp_problem.Graph.start
    end_idx = ipp_problem.Graph.goal
    dist = ipp_problem.Graph.distances
    k_sensors = mipp.k
    timeout = ipp_problem.solution_time
    
    model = Model(Mosek.Optimizer)

    @variable(model, Y[1:M, 1:M])#, PSD)
    @variable(model, 0 <= z[1:N] <= 1)
    
    # this uses the fact that A[:,i]*A[:,i]' is just I since A itself is I
    zaaᵀ = zeros(N)
    zaaᵀ[path] .= 1.0
    
    # this is our posterior from the path which becomes our prior for sample selection
    Σₚ⁻¹ = Σx_inv + Diagonal(σ_max^(-2) .* zaaᵀ)
    
    # Objective
    @objective(model, Min, tr(Y))

    pred_idx = find_closest_points(Theta, Omega)

    # Σ_map_inv = JuMP.@expression(model, (Diagonal(z .* 1/σ_min) + Σₚ⁻¹)[pred_idx, pred_idx])
    Σ_map_inv = JuMP.@expression(model, (Diagonal(z[pred_idx] .* 1/σ_min) + Σₚ⁻¹[pred_idx, pred_idx]))
    @constraint(model, [Y Diagonal(ones(M)); Diagonal(ones(M)) Σ_map_inv] >= 0, PSDCone())

    idx_not_on_path = [i for i in 1:N if i ∉ path]
    @constraint(model, [i in idx_not_on_path], z[i] == 0)
    @constraint(model, sum(z[i] for i in 1:N) == k_sensors)


    set_time_limit_sec(model, timeout)
    optimize!(model)

    if termination_status(model) != OPTIMAL
        @warn "Model not solved to optimality. Terminated with status: $(termination_status(model))"
    end
    if !is_positive_semidefinite(JuMP.value.(Σ_map_inv))
        @warn "Σ_map_inv PSDCone() constraint NOT satisfied. Numerical issues likely encountered."
    end

    # take top k_sensor values from z
    optimal_z = JuMP.value.(model[:z])
    indices_values = [(i, optimal_z[i]) for i in path]
    sorted_indices_values = sort(indices_values, by=x->x[2], rev=true)[1:min(k_sensors, length(indices_values))]
    drills = [tuple[1] for tuple in sorted_indices_values]

    # scatter(Theta[:, 1], Theta[:, 2], label="Theta")
    # Plots.plot!(Theta[path, 1], Theta[path, 2], label="Path", lw=2)
    # scatter!(Theta[pred_idx, 1], Theta[pred_idx, 2], lw=2)
    # scatter!(Omega[:, 1], Omega[:, 2], lw=2)
    # scatter!(Theta[drills, 1], Theta[drills, 2], label="Drills", lw=2)
    # savefig("/Users/joshuaott/Downloads/heatmap.png")

    # Plots.plot([optimal_z[i] for i in path])
    # savefig("/Users/joshuaott/Downloads/ν_inv.png")

    @show sum(optimal_z[i] for i in path)

    return drills
end