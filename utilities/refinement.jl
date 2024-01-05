function run_swap_abstractgp(ipp_problem::IPP, G_dict, path::Vector{Int}, iter::Int)
    n = ipp_problem.n
    m = ipp_problem.m
    start = ipp_problem.Graph.start
    goal = ipp_problem.Graph.goal
    objective = ipp_problem.objective
    Theta = ipp_problem.Graph.Theta
    Omega = ipp_problem.Graph.Omega
    all_pairs_shortest_paths = ipp_problem.Graph.all_pairs_shortest_paths

    if objective != "A-IPP" && objective != "D-IPP"
        error("objective must be either a-optimal or d-optimal")
    end

    if iter > 500
        return path
    end

    for _ in 1:5000
        # location to be swapped
        swap_node = rand(path[2:end-2]) # we can't swap the start or end points
        path_idx = findfirst(x->x==swap_node, path)

        # find neighbors of swap_node
        neighbors = G_dict[swap_node]

        for ni in neighbors
            if ni in path
                continue
            end

            path_to_ni = vcat(path[1:path_idx], shortest_path(all_pairs_shortest_paths, path[path_idx], ni)[2:end])
            ni_to_path = vcat(shortest_path(all_pairs_shortest_paths, ni, path[path_idx+2])[2:end], path[path_idx+2:end][2:end])
            new_path = vcat(path_to_ni, ni_to_path)

            if length(new_path) != length(path)
                # a valid swap only changes node positions but does not add additional nodes to the path
                # this allows for swap to be done efficiently
                continue
            end

            if objective(ipp_problem, new_path) < objective(ipp_problem, path)
                return run_swap_abstractgp(ipp_problem, G_dict, new_path, iter+1)
            end

        end
    end

    return path
end


function run_refinement(data, objective, data_path="/Users/joshuaott/InformativePathPlanning/data")

    if objective != "A-IPP" && objective != "D-IPP"
        error("objective must be either A or D-IPP")
    end

    rng = MersenneTwister(1234567)
    n = 0 # set to dumby to start
    m = data[1].m
    edge_length = 1
    L = 0.01*edge_length # length scale

    refined_paths = []
    refinded_obj_vals = []

    for i in 1:length(data)
        println("$(i)/$(length(data)) | n = $(n)")

        if n != data[i].n
            n = data[i].n
            m = data[i].m
            start = 1
            goal = n

            Graph = build_graph(rng, data_path, n, m, edge_length, start, goal, objective)
            Omega = data.Omega
            Graph = IPPGraph(Graph.G, Graph.start, Graph.goal, Graph.Theta, Omega, Graph.all_pairs_shortest_paths, Graph.distances, Graph.true_map, Graph.edge_length)
            G = Graph.G
            G_dict = Dict(j => Set(G[j]) for j in 1:N)
        end

        path = data[i].path
        σ = data[i].σ
        B = data[i].B
        solution_time = data[i].timeout
        replan_rate = data[i].replan_rate

        # Generate a new measurement model since Omega was updated
        Σₓ = kernel(Graph.Omega, Graph.Omega, L) # = K(X⁺, X⁺)
        Σₓ = round.(Σₓ, digits=8)
        ϵ = Matrix{Float64}(I, size(Σₓ))*1e-6 # Add a Small Constant to the Diagonal (Jitter): This is a common technique to improve the numerical stability of a kernel matrix. 
        Σₓ⁻¹ = inv(Σₓ + ϵ)
        Σₓ⁻¹ = round.(Σₓ⁻¹, digits=8)
        KX⁺X = kernel(Graph.Omega, Graph.Theta, L) # = K(X⁺, X)
        Aᵀ = Σₓ⁻¹ * KX⁺X
        A = Aᵀ'
        A = round.(A, digits=8)
        measurement_model = MeasurementModel(σ, Σₓ, Σₓ⁻¹, L, A)

        # Create an IPP problem
        ipp_problem = IPP(rng, n, m, Graph, measurement_model, objective, B, solution_time, replan_rate)

        swapped_path = run_swap_abstractgp(ipp_problem, G_dict, path, 1)

        refined_obj_val = objective(ipp_problem, swapped_path)

        println(data[i].run_type * " Improvement: ", data[i].objVal - refined_obj_val) 

        push!(refined_paths, swapped_path)
        push!(refinded_obj_vals, refined_obj_val)
    end

    return refined_paths, refinded_obj_vals
end