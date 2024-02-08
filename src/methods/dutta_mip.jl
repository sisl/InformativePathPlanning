function scale(n::Int, lower::Int, upper::Int)
    # Define the input range
    x1 = 4
    x2 = 3844

    # Calculate the corresponding output
    r = (((n - x1) * (upper - lower)) / (x2 - x1)) + lower

    # Round to nearest integer and return
    return round(Int, r)
end

function compute_mip_path(n::Int, G::Vector{Vector{Int64}}, z, start::Int, end_idx::Int, dist::Matrix{Float64})
    v1 = start
    soln = [v1]
    while v1 != end_idx
        idx = argmax([round(JuMP.value(z[(v1, v2)])) for v2 in G[v1]])
        v1 = G[v1][idx]
        push!(soln, v1)
    end

    len = 0.0
    count = 0
    for i in 1:n
        for j in G[i]
            if round(JuMP.value(z[(i, j)])) == 1
                @assert i in soln
                @assert j in soln
                count += 1
            end
            len += dist[i, j] * round(JuMP.value(z[(i, j)]))
        end
    end

    return soln, len
end

function run_dutta_mip(ipp_problem::IPP, idx)
    """
    Solves via the method presented by:  
    Dutta, Shamak, Nils Wilde, and Stephen L. Smith. "Informative Path Planning in Random Fields via Mixed Integer Programming."
    2022 IEEE 61st Conference on Decision and Control (CDC). IEEE, 2022.
    """
    if ipp_problem.solver_type == "open"
        @error("Gurobi solver is required for Dutta et al. method")
    else
        model = Model(Gurobi.Optimizer) # Mosek does not support SOS1 constraints
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

    # NOTE: in Dutta et al. formulation our n is equal to their M and our m is equal to their N
    # here we use our n and m to be consistent with the rest of the code
    C = kernel(ipp_problem.Graph.Theta, ipp_problem.Graph.Theta, ipp_problem.MeasurementModel.L) + ipp_problem.MeasurementModel.σ^2 * Diagonal(ones(n))
    b = kernel(ipp_problem.Graph.Theta, ipp_problem.Graph.Omega, ipp_problem.MeasurementModel.L)
    v = diag(kernel(ipp_problem.Graph.Omega, ipp_problem.Graph.Omega, ipp_problem.MeasurementModel.L))
    w = ones(m)

    # Adjust to numerical precision
    C = round.(C, digits=5)
    b = round.(b, digits=5) 

    # set timer AFTER we have constructed problem for more fair comparison to Dutta et al.
    tick()

    @variable(model, z[idx], Bin)

    # define continuous coefficient variables
    @variable(model, alpha[1:n, 1:m])

    # build quadratic objective
    objs = @expression(model, [i=1:m], w[i] * (sum(alpha[j, i] * C[j, k] * alpha[k, i] for j in 1:n for k in 1:n) - 2 * sum(b[j, i] * alpha[j, i] for j in 1:n) + v[i]))

    # minimize total estimation error
    @objective(model, Min, sum(objs))
    
    @variable(model, dummy[1:n], Bin)
    for i in 1:n
        if i == start || i == goal
            continue
        else
            inter = sum(z[(i, j)] for j in G_dict[i])
            @constraint(model, dummy[i] == 1 - inter)
            for j in 1:m
                @constraint(model, [alpha[i, j], dummy[i]] in SOS1())
            end
        end
    end

    # exactly one edge out of start vertex and one edge into end vertex
    @constraint(model, sum(z[(start, j)] for j in G_dict[start]) == 1)
    @constraint(model, sum(z[(i, goal)] for i in 1:n if goal in G_dict[i]) == 1)

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


    # subtour elimination callback
    function subtour_callback(cb_data, model, G, goal)

        status = callback_node_status(cb_data, model)

        if status != MOI.CALLBACK_NODE_STATUS_INTEGER
            return  # Only run at integer solutions
        else
            sol_z = callback_value.(cb_data, model[:z])
            sol_alpha = callback_value.(cb_data, model[:alpha])

            # get subtours
            N = size(sol_alpha, 1)
            tour, cycles = subtour(sol_z, G, N, goal)
            # push!(tours, cycles)
            if length(tour) > 0 && length(tour) < N-1
                con = @build_constraint(sum(model[:z][(i, j)] for (i,j) in collect(permutations(tour, 2)) if j in G[i]) <= length(tour) - 1)
                MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            end
        end
    end

    # function to check for subtours
    function subtour(sol_z, G, n, goal)
        visited = falses(n)
        min_length_cycle = Inf
        min_cycle_vertices = []
        cycles = []

        for current in 1:n
            singleton = maximum([round(sol_z[(current, i)]) for i in G[current]])

            if !visited[current] && singleton > 0
                cycle_vertices = []
                done = false

                while !done
                    visited[current] = true

                    # find next vertex on path
                    conn = [sol_z[(current, adj)] for adj in G[current]]
                    adjacent = G[current][argmax(conn)]

                    push!(cycle_vertices, current)

                    if current == goal
                        # special case when the end vertex is on the path
                        break
                    end

                    current = adjacent

                    # if the next vertex is visited, we are done
                    done = visited[adjacent]
                end

                push!(cycles, cycle_vertices)

                if length(cycle_vertices) < min_length_cycle && cycle_vertices[end] != goal
                    min_length_cycle = length(cycle_vertices)
                    min_cycle_vertices = cycle_vertices
                end
            end
        end

        return min_cycle_vertices, cycles
    end
    MOI.set(model, MOI.LazyConstraintCallback(), (cb_data) -> subtour_callback(cb_data, model, G, goal))

    setup_time = tok()
    set_time_limit_sec(model, setup_time < timeout ? timeout-setup_time : 0.0)

    JuMP.optimize!(model)

    if termination_status(model) == MOI.INFEASIBLE
        println("Model Infeasible!")
        error("Model Infeasible!")
    end

    z = JuMP.value.(model[:z])
    alpha = JuMP.value.(model[:alpha])

    # Construct solution
    soln, len = compute_mip_path(n, G, z, start, goal, dist)

    obj = objective_value(model)
    runtime = solve_time(model)
   
    # return obj, soln, alpha, z
    return soln, obj
end

function solve(ipp_problem::IPP, method::DuttaMIP)
    """ 
    Takes in IPP problem definition and returns the path and objective value
    using the solution method specified by method.
    """

    idx = []
    for (v1, edges) in collect(enumerate(ipp_problem.Graph.G))
        for v2 in edges
            push!(idx, (v1, v2)) 
        end
    end
        
    if ipp_problem.objective == "A-IPP"
        path, objVal = run_dutta_mip(ipp_problem, idx)

        return path, objective(ipp_problem, path)
    else
        error("Objective not recognized for DuttaMIP")
    end
end