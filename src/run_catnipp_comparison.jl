using JSON
using NPZ
using Gurobi
using MosekTools
using Graphs
using Distances
using AbstractGPs
using Random
using LinearAlgebra
using ProgressMeter

function create_graph_from_dict(graph_dict, node_coords, route, greedy_route)
    graph = Dict("nodes" => Set(graph_dict["nodes"]), "edges" => Dict())

    for (from_node, to_node_edges) in graph_dict["edges"]
        graph["edges"][from_node] = Dict()
        for (to_node, edge_dict) in to_node_edges
            edge = Dict("to_node" => edge_dict["to_node"], "length" => edge_dict["length"])
            graph["edges"][from_node][to_node] = edge
        end
    end

    # Assuming graph["edges"] is your input dictionary
    graph_edges = graph["edges"]

    # Determine the maximum node index, adding 1 to account for 0-indexed nodes
    max_node_index = maximum([parse(Int, key) for key in keys(graph_edges)])

    # Initialize a vector of empty vectors of integers, with size max_node_index + 1
    G = [Int[] for _ in 0:max_node_index]

    # Populate the vector G
    for (from_node, neighbors) in graph_edges
        from_node_idx = parse(Int, from_node)  # Convert node to integer
        for to_node in keys(neighbors)
            push!(G[from_node_idx + 1], parse(Int, to_node)+1)
        end
    end

    # Remove any self neighbors from G (if any)
    for i in 1:length(G)
        G[i] = filter(x -> x != i, G[i])
    end

    # Initialize indices
    start_idx = 2  # Current position of the start node
    goal_idx = 1   # Current position of the goal node
    last_idx = length(G)  # Last index position

    # Step 1: Move the start node (currently at index 2) to index 1
    G[start_idx], G[goal_idx] = G[goal_idx], G[start_idx]
    node_coords[start_idx, :], node_coords[goal_idx, :] = node_coords[goal_idx, :], node_coords[start_idx, :]

    # After the swap, start_idx is now at goal_idx's original position
    # and goal_idx is at start_idx's original position.
    temp_idx = start_idx  # store original start_idx's position
    start_idx = goal_idx  # start_idx is now at goal_idx's original position
    goal_idx = temp_idx   # goal_idx is now at start_idx's original position

    # Update the route list to reflect the swap between start and goal
    for i in 1:length(route)
        if route[i] == temp_idx
            route[i] = start_idx
        elseif route[i] == start_idx
            route[i] = temp_idx
        end
    end
    for i in 1:length(greedy_route)
        if greedy_route[i] == temp_idx
            greedy_route[i] = start_idx
        elseif greedy_route[i] == start_idx
            greedy_route[i] = temp_idx
        end
    end

    # Step 2: Move the goal node (currently at index 1) to the last index
    G[goal_idx], G[last_idx] = G[last_idx], G[goal_idx]
    node_coords[goal_idx, :], node_coords[last_idx, :] = node_coords[last_idx, :], node_coords[goal_idx, :]

    # Update the goal_idx to last_idx after the swap
    for i in 1:length(route)
        if route[i] == goal_idx
            route[i] = last_idx
        elseif route[i] == last_idx
            route[i] = goal_idx
        end
    end
    for i in 1:length(greedy_route)
        if greedy_route[i] == goal_idx
            greedy_route[i] = last_idx
        elseif greedy_route[i] == last_idx
            greedy_route[i] = goal_idx
        end
    end

    # Step 3: Update references in G to reflect new indices
    function update_references!(G, old_idx, new_idx)
        for i in 1:length(G)
            G[i] = [x == old_idx ? new_idx : x == new_idx ? old_idx : x for x in G[i]]
        end
    end

    # Update all references after swaps in the graph G
    update_references!(G, temp_idx, start_idx)  # After swapping start with node at index 1
    update_references!(G, goal_idx, last_idx)   # After moving goal to last index

    # Ensure there are no self-references (optional, depending on your graph's characteristics)
    for i in 1:length(G)
        G[i] = filter(x -> x != i, G[i])
    end

    return graph, G, node_coords, route, greedy_route
end

function run_aspo_catnipp_experiment(node_coords::Matrix{Float64}, ground_truth::Vector{Float64}, G::Vector{Vector{Int}}, budget::Float64)
    rng = Random.MersenneTwister(12345)

    n = size(node_coords, 1)
    objective = "catnipp"
    edge_length = 1
    B = budget
    solution_time = 120.0
    replan_rate = 3
    true_map = reshape(ground_truth, 30, 30)

    Theta = node_coords

    # Generate query locations
    # omega_x = rand(rng, m)*edge_length
    # omega_y = rand(rng, m)*edge_length
    # Omega = hcat(omega_x, omega_y)

    # GRID QUERY LOCATIONS
    x = LinRange(0, 1, 12)
    y = LinRange(0, 1, 12)
    grid = [[[i, j] for i in x, j in y]...]
    Omega = Matrix(hcat(grid...)')
    m = size(Omega, 1)

    # Generate all_pairs_shortest_paths
    graph = build_graph(G, n, Theta)
    all_pairs_shortest_paths = Graphs.floyd_warshall_shortest_paths(graph)

    # Generate the distance matrix
    dist = pairwise(Euclidean(), Theta', dims=2)

    Graph = IPPGraph(G, 1, size(Theta, 1), Theta, Omega, all_pairs_shortest_paths, dist, true_map, edge_length)


    function maternkernel(x::Vector{Vector{Float64}}, y::Vector{Vector{Float64}}, L, σ_0=1)
        matern_kernel = with_lengthscale(MaternKernel(ν=1.5), L)
        K = kernelmatrix(matern_kernel, x, y)
        return σ_0 * K
    end

    function maternkernel(x::Matrix{Float64}, y::Matrix{Float64}, L, σ_0=1)
        matern_kernel = with_lengthscale(MaternKernel(ν=1.5), L)
        K = kernelmatrix(matern_kernel, x', y')
        return σ_0 * K
    end

    # NOTE: catnipp uses sklearn GaussianProcessRegressor with alpha set to default of 1e-10
    # but 1e-10 causes positive definite matrix error in Julia. So, we use 1e-7 instead.
    σ = 1e-7
    L = 0.45*edge_length # length scale 
    Σₓ = maternkernel(Graph.Omega, Graph.Omega, L) # = K(X⁺, X⁺)
    Σₓ = round.(Σₓ, digits=8)
    ϵ = Matrix{Float64}(I, size(Σₓ))*1e-6 # Add a Small Constant to the Diagonal (Jitter): This is a common technique to improve the numerical stability of a kernel matrix. 
    Σₓ⁻¹ = inv(Σₓ + ϵ)
    Σₓ⁻¹ = round.(Σₓ⁻¹, digits=8)
    KX⁺X = maternkernel(Graph.Omega, Graph.Theta, L) # = K(X⁺, X)
    Aᵀ = Σₓ⁻¹ * KX⁺X
    A = Aᵀ'
    A = round.(A, digits=8)

    measurement_model = MeasurementModel(σ, Σₓ, Σₓ⁻¹, L, A)

    # Create an IPP problem
    ipp_problem = IPP(rng, n, m, Graph, measurement_model, objective, B, solution_time, replan_rate, "commercial")

    # Solve the IPP problem
    # val, t = @timed solve(ipp_problem, Exact())
    # val, t = @timed solve(ipp_problem, trΣ⁻¹())
    val, t = @timed solve(ipp_problem, ASPOCAtNIPPComparison())
    path, objective_value = val

    return path, ipp_problem
end

function local_opt(ipp_problem::IPP, G_dict, path::Vector{Int}, iter::Int, start_time::Float64)
    n = ipp_problem.n
    m = ipp_problem.m
    start = ipp_problem.Graph.start
    goal = ipp_problem.Graph.goal
    obj = ipp_problem.objective
    Theta = ipp_problem.Graph.Theta
    Omega = ipp_problem.Graph.Omega
    all_pairs_shortest_paths = ipp_problem.Graph.all_pairs_shortest_paths

    if obj != "A-IPP" && obj != "D-IPP" && obj != "catnipp"
        error("objective must be either a-optimal or d-optimal or catnipp")
    end

    if iter > 500
        return path
    end

    for _ in 1:100
        if time() - start_time > 15
            println("reached time limit")
            return path
        end
        # location to be swapped
        swap_node = rand(path[2:end-2]) # we can't swap the start or end points
        path_idx = findfirst(x->x==swap_node, path)

        # find neighbors of swap_node
        neighbors = G_dict[swap_node]

        for ni in neighbors
            if ni in path
                continue
            end

            path_to_ni = vcat(path[1:path_idx], InformativePathPlanning.shortest_path(all_pairs_shortest_paths, path[path_idx], ni)[2:end])
            ni_to_path = vcat(InformativePathPlanning.shortest_path(all_pairs_shortest_paths, ni, path[path_idx+2])[2:end], path[path_idx+2:end][2:end])
            new_path = vcat(path_to_ni, ni_to_path)

            if InformativePathPlanning.path_distance(ipp_problem, new_path) > ipp_problem.B
                continue
            end

            if InformativePathPlanning.catnipp_objective(ipp_problem, new_path) < InformativePathPlanning.catnipp_objective(ipp_problem, path)
                return local_opt(ipp_problem, G_dict, new_path, iter+1, start_time)
            end

        end
    end

    return path
end


function get_measurement(node_coord::Vector{Float64}, true_map::Matrix{Float64})
    x = LinRange(0, 1, 30)
    y = LinRange(0, 1, 30)
    grid = [[[i, j] for i in x, j in y]...]
    true_map_node_coords = Matrix(hcat(grid...)')

    closest_node = argmin([norm(node_coord - true_map_node_coords[i, :]) for i in 1:size(true_map_node_coords, 1)])
    
    return true_map[closest_node]
end

function get_measurement(node_coord::Matrix{Float64}, true_map::Matrix{Float64})
    x = LinRange(0, 1, 30)
    y = LinRange(0, 1, 30)
    grid = [[[i, j] for i in x, j in y]...]
    true_map_node_coords = Matrix(hcat(grid...)')

    closest_node = argmin([norm(node_coord' - true_map_node_coords[i, :]) for i in 1:size(true_map_node_coords, 1)])
    
    return true_map[closest_node]
end

function get_measurement_history(path::Vector{Int64}, true_map::Matrix{Float64}, node_coords::Matrix{Float64})
    x = LinRange(0, 1, 30)
    y = LinRange(0, 1, 30)
    grid = [[[i, j] for i in x, j in y]...]
    true_map_node_coords = Matrix(hcat(grid...)')

    y_hist = Vector{Float64}()

    for i in 1:length(path)
        node_coord = node_coords[path[i], :]
        closest_node = argmin([norm(node_coord - true_map_node_coords[i, :]) for i in 1:size(true_map_node_coords, 1)])
        push!(y_hist, true_map[closest_node])
    end

    return y_hist

end

function query_high_interest(true_map::Matrix{Float64}, Theta::Matrix{Float64}, path::Vector{Int64}, L=0.45, σ=1e-7)
    # computes the variange in high interest areas 
    y_hist = get_measurement_history(path, true_map, Theta)

    gp = AbstractGPs.GP(with_lengthscale(MaternKernel(ν=1.5), L))
    # create grid of 30x30 points in [0,1]x[0,1]
    x = LinRange(0, 1, 30)
    y = LinRange(0, 1, 30)
    grid = [[[i, j] for i in x, j in y]...]
    Ω = grid

    ν = σ^2 .* ones(1:length(path))

    x = Theta[path, :]
    X = [x[i, :] for i in 1:size(x, 1)]    
    y = y_hist
    post_gp = AbstractGPs.posterior(gp(X, ν), y)

    β = 1 
    μ = mean(post_gp(Ω))
    std = sqrt.(var(post_gp(Ω)))
    # This uses the resulting GP belief to determine the high interest nodes
    high_interest_idxs = [(μ + β * std)[i] > 0.4 ? 1 : 0 for i in 1:length(Ω)]
    high_interest_grid = zeros(900)
    high_interest_grid[high_interest_idxs .== 1] .= 1
    return reshape(high_interest_grid, 30, 30)
end

function catnipp_objective(ipp_problem::IPP, path::Vector{Int64})
    # computes the variange in high interest areas 
    y_hist = get_measurement_history(path, ipp_problem.Graph.true_map, ipp_problem.Graph.Theta) #ipp_problem.Graph.true_map[path]

    gp = AbstractGPs.GP(with_lengthscale(MaternKernel(ν=1.5), ipp_problem.MeasurementModel.L))
    # create grid of 30x30 points in [0,1]x[0,1]
    x = LinRange(0, 1, 30)
    y = LinRange(0, 1, 30)
    grid = [[[i, j] for i in x, j in y]...]
    Ω = grid

    ν = ipp_problem.MeasurementModel.σ^2 .* ones(1:length(path))

    x = ipp_problem.Graph.Theta[path, :]
    X = [x[i, :] for i in 1:size(x, 1)]    
    y = y_hist
    post_gp = AbstractGPs.posterior(gp(X, ν), y)

    β = 1 
    μ = mean(post_gp(Ω))
    std = sqrt.(var(post_gp(Ω)))
    # This uses the resulting GP belief to determine the high interest nodes
    # high_interest_nodes = Ω[findall(μ + β * std .> 0.4)]
    
    # What we actually care about is the variance in the ground truth high interest nodes
    high_interest_idxs = [ipp_problem.Graph.true_map[i] > 0.4 ? 1 : 0 for i in 1:900]
    high_interest_nodes = Ω[high_interest_idxs .== 1]

    if isempty(high_interest_nodes)
        println("No high interest nodes found")
        return Inf
    else
        variances = var(post_gp(high_interest_nodes))
        return sum(variances)
    end
end

function posterior_estimate(path::Vector{Int64}, query_size::Tuple{Int, Int}, Theta::Matrix{Float64}, true_map, L=0.45, σ=1e-7)
    y_hist = get_measurement_history(path, true_map, Theta)

    gp = AbstractGPs.GP(with_lengthscale(MaternKernel(ν=1.5), L))
    # create grid of 30x30 points in [0,1]x[0,1]
    x = LinRange(0, 1, query_size[1])
    y = LinRange(0, 1, query_size[2])
    grid = [[[i, j] for i in x, j in y]...]
    Ω = grid

    ν = σ^2 .* ones(1:length(path))

    x = Theta[path, :]
    X = [x[i, :] for i in 1:size(x, 1)]    
    y = y_hist
    post_gp = AbstractGPs.posterior(gp(X, ν), y)

    variances = var(post_gp(Ω))
    μ = mean(post_gp(Ω))
    return μ, variances
end

function compute_obj_hist(ipp_problem::IPP, path::Vector{Int64})
    true_map = ipp_problem.Graph.true_map
    obj_hist = []
    for i in 2:length(path)
        push!(obj_hist, catnipp_objective(ipp_problem, path[1:i]))
    end
    return obj_hist
end

function compute_budget_hist(ipp_problem::IPP, path::Vector{Int64})
    budget_hist = []
    for i in 1:length(path)
        push!(budget_hist, InformativePathPlanning.path_distance(ipp_problem, path[1:i]))
    end
    return budget_hist
end

function check_path_distance(ipp_problem::IPP, path::Vector{Int64})
    if InformativePathPlanning.path_distance(ipp_problem, path) > ipp_problem.B
        # we want to start from the back of the path and look at removing that segment to the goal and replacing it with the shortest path to the goal until we are under the budget constraint
        for i in length(path):-1:1
            temp_path = vcat(path[1:i-1], InformativePathPlanning.shortest_path(ipp_problem.Graph.all_pairs_shortest_paths, path[i], ipp_problem.Graph.goal))
            if InformativePathPlanning.path_distance(ipp_problem, temp_path) <= ipp_problem.B
                return temp_path
            end
        end
    else
        return path
    end
end


"""
To run the comparison between ASPO and CATNIPP, run the following function 
after you have run the CATNIPP experiments and saved the graph, node_coords,
ground_truth, and route data. This function will then run the ASPO experiments
on the same graph and save the results for comparison. 
https://github.com/marmotlab/CAtNIPP/tree/main
"""
function run_catnipp_comparison()
    data_lock = ReentrantLock()

    data_path = joinpath(@__DIR__, "..", "data")

    @showprogress dt=0.5 for budget in [4, 6, 8, 10, 12]
        aspo_paths = []
        aspo_obj_hists = []
        aspo_budget_hists = []
        aspo_distances = []
        aspo_objectives = []
        aspo_planning_times = []
        catnipp_ts_obj_hists = []
        catnipp_ts_budget_hists = []
        catnipp_ts_distances = []
        catnipp_ts_objectives = []
        catnipp_greedy_obj_hists = []
        catnipp_greedy_budget_hists = []
        catnipp_greedy_distances = []
        catnipp_greedy_objectives = []

        Threads.@threads for i in 0:99
            graph_data = JSON.parsefile(data_path * "/catnipp_results/budget_$(budget)/graph/$(i)_graph.json")
            node_coords = npzread(data_path * "/catnipp_results/budget_$(budget)/graph/$(i)_node_coords.npy")
            ground_truth = npzread(data_path * "/catnipp_results/budget_$(budget)/graph/$(i)_ground_truth.npy")
            route = npzread(data_path * "/catnipp_results/budget_$(budget)/routes/$(i)_route.npy")
            greedy_route = npzread(data_path * "/catnipp_results/budget_$(budget)/greedy/routes/$(i)_route.npy")
            route = route .+ 1 # Convert to 1-indexed
            greedy_route = greedy_route .+ 1 # Convert to 1-indexed

            # Create the graph from the dictionary
            graph, G, node_coords, route, greedy_route = create_graph_from_dict(graph_data, node_coords, route, greedy_route)

            val, t = @timed run_aspo_catnipp_experiment(node_coords, ground_truth, G, float(budget))
            path, ipp_problem = val

            path = check_path_distance(ipp_problem, path)

            # G_dict = Dict(j => Set(G[j]) for j in 1:ipp_problem.n)
            # local_opt_path = local_opt(ipp_problem, G_dict, path, 1, time())

            lock(data_lock) do
                aspo_paths = push!(aspo_paths, path)
                aspo_distances = push!(aspo_distances, InformativePathPlanning.path_distance(ipp_problem, path))
                aspo_objectives = push!(aspo_objectives, catnipp_objective(ipp_problem, path))
                aspo_obj_hists = push!(aspo_obj_hists, compute_obj_hist(ipp_problem, path))
                aspo_budget_hists = push!(aspo_budget_hists, compute_budget_hist(ipp_problem, path))
                aspo_planning_times = push!(aspo_planning_times, t)
                catnipp_ts_distances = push!(catnipp_ts_distances, InformativePathPlanning.path_distance(ipp_problem, route))
                catnipp_ts_objectives = push!(catnipp_ts_objectives, catnipp_objective(ipp_problem, route))
                catnipp_ts_obj_hists = push!(catnipp_ts_obj_hists, compute_obj_hist(ipp_problem, route))
                catnipp_ts_budget_hists = push!(catnipp_ts_budget_hists, compute_budget_hist(ipp_problem, route))
                catnipp_greedy_distances = push!(catnipp_greedy_distances, InformativePathPlanning.path_distance(ipp_problem, greedy_route))
                catnipp_greedy_objectives = push!(catnipp_greedy_objectives, catnipp_objective(ipp_problem, greedy_route))
                catnipp_greedy_obj_hists = push!(catnipp_greedy_obj_hists, compute_obj_hist(ipp_problem, greedy_route))
                catnipp_greedy_budget_hists = push!(catnipp_greedy_budget_hists, compute_budget_hist(ipp_problem, greedy_route))

                println("ASPO Objective: $(mean(aspo_objectives))")
                println("CatNIPP TS Objective: $(mean(catnipp_ts_objectives))")
                println("CatNIPP Greedy Objective: $(mean(catnipp_greedy_objectives))")
                println("ASPO planning time: $(mean(aspo_planning_times))")
            end
        end

        # Save the results with JLD2
        JLD2.save(data_path * "/catnipp_results/budget_$(budget)/aspo_paths.jld2", "aspo_paths", aspo_paths)
        JLD2.save(data_path * "/catnipp_results/budget_$(budget)/aspo_distances.jld2", "aspo_distances", aspo_distances)
        JLD2.save(data_path * "/catnipp_results/budget_$(budget)/aspo_objectives.jld2", "aspo_objectives", aspo_objectives)
        JLD2.save(data_path * "/catnipp_results/budget_$(budget)/aspo_obj_hists.jld2", "aspo_obj_hists", aspo_obj_hists)
        JLD2.save(data_path * "/catnipp_results/budget_$(budget)/aspo_budget_hists.jld2", "aspo_budget_hists", aspo_budget_hists)
        JLD2.save(data_path * "/catnipp_results/budget_$(budget)/aspo_planning_times.jld2", "aspo_planning_times", aspo_planning_times)
        JLD2.save(data_path * "/catnipp_results/budget_$(budget)/catnipp_ts_distances.jld2", "catnipp_ts_distances", catnipp_ts_distances)
        JLD2.save(data_path * "/catnipp_results/budget_$(budget)/catnipp_ts_objectives.jld2", "catnipp_ts_objectives", catnipp_ts_objectives)
        JLD2.save(data_path * "/catnipp_results/budget_$(budget)/catnipp_ts_obj_hists.jld2", "catnipp_ts_obj_hists", catnipp_ts_obj_hists)
        JLD2.save(data_path * "/catnipp_results/budget_$(budget)/catnipp_ts_budget_hists.jld2", "catnipp_ts_budget_hists", catnipp_ts_budget_hists)
        JLD2.save(data_path * "/catnipp_results/budget_$(budget)/catnipp_greedy_distances.jld2", "catnipp_greedy_distances", catnipp_greedy_distances)
        JLD2.save(data_path * "/catnipp_results/budget_$(budget)/catnipp_greedy_objectives.jld2", "catnipp_greedy_objectives", catnipp_greedy_objectives)
        JLD2.save(data_path * "/catnipp_results/budget_$(budget)/catnipp_greedy_obj_hists.jld2", "catnipp_greedy_obj_hists", catnipp_greedy_obj_hists)
        JLD2.save(data_path * "/catnipp_results/budget_$(budget)/catnipp_greedy_budget_hists.jld2", "catnipp_greedy_budget_hists", catnipp_greedy_budget_hists)

    end

end