##########################################################################
# A-IPP
##########################################################################
function run_AIPP_exact(poiipp::PointsOfInterestIPP, idx, relax::Bool=false)
    ipp_problem = poiipp.ipp_problem
    noi = poiipp.noi
    tick()

    if relax
        if ipp_problem.solver_type == "open"
            println("Hypatia")
            model = Model(Hypatia.Optimizer)
        else
            if isdefined(Main, :mosek_available)
                println("Mosek")
                model = Model(Mosek.Optimizer)
            else
                @error("Commercial solver specified but not available.")
            end
        end
    else
        if ipp_problem.solver_type == "open"
            println("HiGHS & Hypatia")
            model = Model(optimizer_with_attributes(
                    Pajarito.Optimizer,
                    "oa_solver" => optimizer_with_attributes(
                        HiGHS.Optimizer,
                        MOI.Silent() => true,
                        "mip_feasibility_tolerance" => 1e-8,
                        "mip_rel_gap" => 1e-6,
                    ),
                    "conic_solver" =>
                        optimizer_with_attributes(Hypatia.Optimizer, MOI.Silent() => true),
                )
            )
        else
            if isdefined(Main, :gurobi_available) && isdefined(Main, :mosek_available)
                println("Gurobi & Mosek")
                
                model = Model(
                    optimizer_with_attributes(
                        Pajarito.Optimizer,
                        "oa_solver" => optimizer_with_attributes(
                            Gurobi.Optimizer,
                            MOI.Silent() => true,
                            "FeasibilityTol" => 1e-8,
                            "MIPGap" => 1e-6,
                        ),
                        "conic_solver" =>
                            optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true),
                    )
                )            
            else
                @error("Commercial solver specified but not available.")
            end
        end
    end

    G = ipp_problem.Graph.G
    A = ipp_problem.MeasurementModel.A
    n = ipp_problem.n
    m = ipp_problem.m
    σ = ipp_problem.MeasurementModel.σ
    Σₓ⁻¹ = ipp_problem.MeasurementModel.Σₓ⁻¹
    start = ipp_problem.Graph.start
    goal = ipp_problem.Graph.goal
    B = ipp_problem.B
    timeout = ipp_problem.solution_time
    all_pairs_shortest_paths = ipp_problem.Graph.all_pairs_shortest_paths
    dist = ipp_problem.Graph.distances
    G_dict = Dict(j => Set(G[j]) for j in 1:n)

    @variable(model, Y[1:m, 1:m], PSD)

    if relax
        @variable(model, 0 <= z[idx] <= 1)  
    else
        @variable(model, z[idx], Bin)  
    end

    # Objective
    @objective(model, Min, tr(Y))

    z_sums = [sum(z[(i,j)] for j in G_dict[i]) for i in 1:n]
    ZA_mat = σ^(-2)*sum(z_sums[i]*A[i, :]*A[i, :]' for i in 1:n)
    Σ_est_inv = JuMP.@expression(model, (ZA_mat + Σₓ⁻¹))

    @constraint(model, [Y Diagonal(ones(m)); Diagonal(ones(m)) Σ_est_inv] >= 0, PSDCone())

    # exactly one edge out of start vertex and one edge into end vertex
    @constraint(model, sum(z[(start, j)] for j in G_dict[start]) == 1)
    @constraint(model, sum(z[(i, goal)] for i in G_dict[goal]) == 1)

    # no incoming edge into start, no outgoing edge from end
    for i in 1:n
        for j in G_dict[i]
            if j == start
                @constraint(model, z[(i, j)] == 0)
            end
            if i == goal
                @constraint(model, z[(i, j)] == 0)
            end
        end
    end

    # in-degree == out_degree <= 1
    # Create a dictionary mapping from i to the set of j's for which i in G_dict[j]
    i_to_j = Dict(i => Set() for i in 1:n)
    for j in 1:n
        for i in G_dict[j]
            push!(i_to_j[i], j)
        end
    end
    
    for i in 1:n
        if i != start && i != goal
            # Use the pre-calculated mapping to directly iterate over the relevant j values
            second_sum = sum(z[(j, i)] for j in i_to_j[i])
    
            @constraint(model, sum(z[(i, j)] for j in G_dict[i]) == second_sum)
            @constraint(model, second_sum <= 1)
        end
    end
    
    # path length constraint
    @constraint(model, sum(dist[i, j] * z[(i, j)] for i in 1:(n-1) for j in G_dict[i]) <= B)

    @variable(model, u[1:n])
    @constraint(model, u[1] == 1)
    @constraint(model, [i=2:n], 2 <= u[i] <= n)
    for i in 2:n
        for j in G_dict[i]
            if i != j
                @constraint(model, u[i] - u[j] + 1 <= (n-1)*(1-z[(i,j)]))
            end
        end
    end

    # nodes of interest constraint
    for i in noi
        @constraint(model, sum(z[(i, j)] for j in G_dict[i]) == 1)
    end

    setup_time = tok()
    
    set_time_limit_sec(model, setup_time < timeout ? timeout-setup_time : 0.0)
    optimize!(model)
    objVal = objective_value(model)
    if objVal == Inf
        @error("No Solution")
    end

    if relax
        return [], objVal
    else
        optimal_z = JuMP.value.(z)
        optimal_u = JuMP.value.(u)
        path = extract_path(ipp_problem, G, deepcopy(optimal_z), deepcopy(optimal_u), start, goal, all_pairs_shortest_paths, dist, B)
        return path, objVal
    end
end

##########################################################################
# D-IPP
##########################################################################
function run_DIPP_exact(poiipp::PointsOfInterestIPP, idx, relax::Bool=false)
    ipp_problem = poiipp.ipp_problem
    noi = poiipp.noi
    tick()

    if relax
        if ipp_problem.solver_type == "open"
            println("Hypatia")
            model = Model(Hypatia.Optimizer)
        else
            if isdefined(Main, :mosek_available)
                println("Mosek")
                model = Model(Mosek.Optimizer)
            else
                @error("Commercial solver specified but not available.")
            end
        end
    else
        if ipp_problem.solver_type == "open"
            println("HiGHS & Hypatia")
            model = Model(optimizer_with_attributes(
                    Pajarito.Optimizer,
                    "oa_solver" => optimizer_with_attributes(
                        HiGHS.Optimizer,
                        MOI.Silent() => true,
                        "mip_feasibility_tolerance" => 1e-8,
                        "mip_rel_gap" => 1e-6,
                    ),
                    "conic_solver" =>
                        optimizer_with_attributes(Hypatia.Optimizer, MOI.Silent() => true),
                )
            )
        else
            if isdefined(Main, :gurobi_available) && isdefined(Main, :mosek_available)
                println("Gurobi & Mosek")

                model = Model(
                    optimizer_with_attributes(
                        Pajarito.Optimizer,
                        "oa_solver" => optimizer_with_attributes(
                            Gurobi.Optimizer,
                            MOI.Silent() => true,
                            "FeasibilityTol" => 1e-8,
                            "MIPGap" => 1e-6,
                        ),
                        "conic_solver" =>
                            optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true),
                    )
                )            
            else
                @error("Commercial solver specified but not available.")
            end

            
        end
    end

    G = ipp_problem.Graph.G
    A = ipp_problem.MeasurementModel.A
    n = ipp_problem.n
    m = ipp_problem.m
    σ = ipp_problem.MeasurementModel.σ
    Σₓ⁻¹ = ipp_problem.MeasurementModel.Σₓ⁻¹
    start = ipp_problem.Graph.start
    goal = ipp_problem.Graph.goal
    B = ipp_problem.B
    timeout = ipp_problem.solution_time
    all_pairs_shortest_paths = ipp_problem.Graph.all_pairs_shortest_paths
    dist = ipp_problem.Graph.distances
    G_dict = Dict(j => Set(G[j]) for j in 1:n)

    if relax 
        @variable(model, 0 <= z[idx] <= 1)  
    else
        @variable(model, z[idx], Bin)  
    end

    @variable(model, t)

    # Objective
    @objective(model, Max, t)

    z_sums = [sum(z[(i,j)] for j in G_dict[i]) for i in 1:n]
    ZA_mat = σ^(-2)*sum(z_sums[i]*A[i, :]*A[i, :]' for i in 1:n)
    Σ_est_inv = JuMP.@expression(model, (ZA_mat + Σₓ⁻¹))

    @constraint(model, [t, 1, (Σ_est_inv[i, j] for i in 1:m for j in 1:i)...] in MOI.LogDetConeTriangle(m))

    # exactly one edge out of start vertex and one edge into end vertex
    @constraint(model, sum(z[(start, j)] for j in G_dict[start]) == 1)
    @constraint(model, sum(z[(i, goal)] for i in G_dict[goal]) == 1)

    # no incoming edge into start, no outgoing edge from end
    for i in 1:n
        for j in G_dict[i]
            if j == start
                @constraint(model, z[(i, j)] == 0)
            end
            if i == goal
                @constraint(model, z[(i, j)] == 0)
            end
        end
    end

    # in-degree == out_degree <= 1
    # Create a dictionary mapping from i to the set of j's for which i in G_dict[j]
    i_to_j = Dict(i => Set() for i in 1:n)
    for j in 1:n
        for i in G_dict[j]
            push!(i_to_j[i], j)
        end
    end
    
    for i in 1:n
        if i != start && i != goal
            # Use the pre-calculated mapping to directly iterate over the relevant j values
            second_sum = sum(z[(j, i)] for j in i_to_j[i])
    
            @constraint(model, sum(z[(i, j)] for j in G_dict[i]) == second_sum)
            @constraint(model, second_sum <= 1)
        end
    end
    
    # path length constraint
    @constraint(model, sum(dist[i, j] * z[(i, j)] for i in 1:(n-1) for j in G_dict[i]) <= B)

    @variable(model, u[1:n])
    @constraint(model, u[1] == 1)
    @constraint(model, [i=2:n], 2 <= u[i] <= n)
    for i in 2:n
        for j in G_dict[i]
            if i != j
                @constraint(model, u[i] - u[j] + 1 <= (n-1)*(1-z[(i,j)]))
            end
        end
    end

    # nodes of interest constraint
    for i in noi
        @constraint(model, sum(z[(i, j)] for j in G_dict[i]) == 1)
    end

    setup_time = tok()
    
    set_time_limit_sec(model, setup_time < timeout ? timeout-setup_time : 0.0)
    optimize!(model)

    # minimize logdet(Σ_est) = maximize -logdet(Σ_est) = maximize logdet(Σ_est_inv)
    # the true objVal we care about is the negative of the one returned by the solver
    # i.e. objVal ≈ logdet(inv(Matrix(JuMP.value.(Σ_est_inv))))
    objVal = objective_value(model)
    objVal = -objVal

    if objVal == Inf || objVal == -Inf
        @error("No Solution")
    end

    if relax
        return [], objVal
    else
        optimal_z = JuMP.value.(z)
        optimal_u = JuMP.value.(u)
        path = extract_path(ipp_problem, G, deepcopy(optimal_z), deepcopy(optimal_u), start, goal, all_pairs_shortest_paths, dist, B)
        
        return path, objVal
    end
end

##########################################################################
# B-IPP
##########################################################################
function run_BIPP_exact(poiipp::PointsOfInterestIPP, idx)
    ipp_problem = poiipp.ipp_problem
    noi = poiipp.noi
    tick()

    if ipp_problem.solver_type == "open"
        println("HiGHS")
        model = Model(HiGHS.Optimizer)
    else
        if isdefined(Main, :gurobi_available)
            println("Gurobi")
            model = Model(Gurobi.Optimizer) # Mosek does not support SOS1 constraints
        else
            @error("Commercial solver specified but not available.")
        end
    end

    G = ipp_problem.Graph.G
    A = ipp_problem.MeasurementModel.A
    n = ipp_problem.n
    m = ipp_problem.m
    σ = ipp_problem.MeasurementModel.σ
    Σₓ⁻¹ = ipp_problem.MeasurementModel.Σₓ⁻¹
    start = ipp_problem.Graph.start
    goal = ipp_problem.Graph.goal
    B = ipp_problem.B
    timeout = ipp_problem.solution_time
    all_pairs_shortest_paths = ipp_problem.Graph.all_pairs_shortest_paths
    dist = ipp_problem.Graph.distances
    G_dict = Dict(j => Set(G[j]) for j in 1:n)

    @variable(model, z[idx], Bin)  

    z_sums = [sum(z[(i,j)] for j in G_dict[i]) for i in 1:n]
    ZA_mat = σ^(-2)*sum(z_sums[i]*A[i, :]*A[i, :]' for i in 1:n)
    Σ_est_inv = JuMP.@expression(model, (ZA_mat + Σₓ⁻¹))

    @objective(model, Max, tr(Σ_est_inv))

    # exactly one edge out of start vertex and one edge into end vertex
    @constraint(model, sum(z[(start, j)] for j in G_dict[start]) == 1)
    @constraint(model, sum(z[(i, goal)] for i in G_dict[goal]) == 1)

    # no incoming edge into start, no outgoing edge from end
    for i in 1:n
        for j in G_dict[i]
            if j == start
                @constraint(model, z[(i, j)] == 0)
            end
            if i == goal
                @constraint(model, z[(i, j)] == 0)
            end
        end
    end

    # in-degree == out_degree <= 1
    # Create a dictionary mapping from i to the set of j's for which i in G_dict[j]
    i_to_j = Dict(i => Set() for i in 1:n)
    for j in 1:n
        for i in G_dict[j]
            push!(i_to_j[i], j)
        end
    end
    
    for i in 1:n
        if i != start && i != goal
            # Use the pre-calculated mapping to directly iterate over the relevant j values
            second_sum = sum(z[(j, i)] for j in i_to_j[i])
    
            @constraint(model, sum(z[(i, j)] for j in G_dict[i]) == second_sum)
            @constraint(model, second_sum <= 1)
        end
    end
    
    # path length constraint
    @constraint(model, sum(dist[i, j] * z[(i, j)] for i in 1:(n-1) for j in G_dict[i]) <= B)

    @variable(model, u[1:n])
    @constraint(model, u[1] == 1)
    @constraint(model, [i=2:n], 2 <= u[i] <= n)
    for i in 2:n
        for j in G_dict[i]
            if i != j
                @constraint(model, u[i] - u[j] + 1 <= (n-1)*(1-z[(i,j)]))
            end
        end
    end

    # nodes of interest constraint
    for i in noi
        @constraint(model, sum(z[(i, j)] for j in G_dict[i]) == 1)
    end

    setup_time = tok()
    set_time_limit_sec(model, setup_time < timeout ? timeout-setup_time : 0.0)
    optimize!(model)

    objVal = objective_value(model)

    if objVal == Inf
        @error("No Solution")
    end

    optimal_z = JuMP.value.(z)
    optimal_u = JuMP.value.(u)
    path = extract_path(ipp_problem, G, deepcopy(optimal_z), deepcopy(optimal_u), start, goal, all_pairs_shortest_paths, dist, B)
    
    return path, objVal
end


##########################################################################
# Solve
##########################################################################
function solve(poiipp::PointsOfInterestIPP, method::SolutionMethod)
    """ 
    Takes in PointsOfInterestIPP problem definition and returns the path and objective value
    using the solution method specified by method.
    """
    if method != trΣ⁻¹() || method != Exact()
        @error("You tried to specify points of interest for a method that is not implemented")
    end

    path, objVal = solve(poiipp, method)

    return path, objVal
end

function solve(poiipp::PointsOfInterestIPP, method::Exact, relax::Bool=false)
    """ 
    Takes in PointsOfInterestIPP problem definition and returns the path and objective value
    using the solution method specified by method.
    """
    ipp_problem = poiipp.ipp_problem

    idx = []
    for (v1, edges) in collect(enumerate(ipp_problem.Graph.G))
        for v2 in edges
            push!(idx, (v1, v2)) 
        end
    end

    if ipp_problem.objective == "A-IPP"
        path, objVal = run_AIPP_exact(poiipp, idx, relax)
        if relax
            return [], objVal
        else
            return path, objective(ipp_problem, path)
        end
    elseif ipp_problem.objective == "B-IPP"
        @error("B-IPP exact method should be called using trΣ⁻¹()")

    elseif ipp_problem.objective == "D-IPP"
        path, objVal = run_DIPP_exact(poiipp, idx, relax)
        if relax
            return [], objVal
        else
            return path, objective(ipp_problem, path)
        end
    else
        error("Objective not recognized Exact")
    end
end

function solve(poiipp::PointsOfInterestIPP, method::trΣ⁻¹, relax::Bool=false)
    """ 
    Takes in PointsOfInterestIPP problem definition and returns the path and objective value
    using the solution method specified by method.
    """
    ipp_problem = poiipp.ipp_problem

    idx = []
    for (v1, edges) in collect(enumerate(ipp_problem.Graph.G))
        for v2 in edges
            push!(idx, (v1, v2)) 
        end
    end

    if ipp_problem.objective == "A-IPP"
        path, objVal = run_BIPP_exact(poiipp, idx)
        return path, objective(ipp_problem, path)

    elseif ipp_problem.objective == "B-IPP"
        path, objVal = run_BIPP_exact(poiipp, idx)
        return path, objective(ipp_problem, path)

    elseif ipp_problem.objective == "D-IPP"
        path, objVal = run_DIPP_exact(poiipp, idx)
        return path, objective(ipp_problem, path)

    else
        error("Objective not recognized trΣ⁻¹")
    end
end