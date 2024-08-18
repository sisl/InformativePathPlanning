function aspo_catnipp_estimate_rewards(ipp_problem::IPP, gp::AbstractGPs.PosteriorGP, path::Vector{Int64}, y_hist::Vector{Float64})
    """
    Estimates the rewards for all reachable nodes in the graph given the current path and the GP. 
    This assumes we can teleport to any node in the graph (i.e. we can visit any node in the graph without having to visit all nodes in the path to get there).
    We use the GP to estimate the value of each node in the graph depending on the problem objective. 
    """

    n = ipp_problem.n
    reachable_nodes = Vector{Int64}()
    rewards = -1e9 .* ones(n)
    Ω = [ipp_problem.Graph.Omega[i, :] for i in 1:size(ipp_problem.Graph.Omega, 1)]

    if ipp_problem.objective == "catnipp"
        # precompute the high interest nodes
        β = 1.0
        μ = mean(gp(Ω))
        std = sqrt.(var(gp(Ω)))
        high_interest_nodes = Ω[findall(μ + β * std .> 0.4)]
    end

    for i in 1:n
        if i ∈ path || ipp_problem.Graph.G[i] == [] # second part is in the case of obstacles in the graph 
            continue
        end

        # NOTE: we look at the path to the node, and then the path from the node to the goal
        # only to check node i is reachable. We only append node i to the current path for the value estimate (not the path to the node)
        # i.e. this is the teleporation assumption in the paper
        path_to_node = vcat(vcat(path[1:end-1], shortest_path(ipp_problem.Graph.all_pairs_shortest_paths, path[end], i)), shortest_path(ipp_problem.Graph.all_pairs_shortest_paths, i, ipp_problem.Graph.goal)[2:end])

        if path_distance(ipp_problem, path_to_node) <= ipp_problem.B
            push!(reachable_nodes, i)

            x = ipp_problem.Graph.Theta[[i], :]
            X = [x[i, :] for i in 1:size(x, 1)]
            # use gp mean to predict value of candidate point
            y = mean(gp([ipp_problem.Graph.Theta[i, :]]))
            post_gp = AbstractGPs.posterior(gp(X, ipp_problem.MeasurementModel.σ^2), y)
            if ipp_problem.objective == "A-IPP"
                # NOTE: this uses post_gp and not gp since we're asking what is the variance at the prediction points IF we vist the candidate point?
                # we want to minimize variance, so we want to go to locations that have greater (less negative) -sum(variance)
                variances = var(post_gp(Ω))
                rewards[i] = -sum(variances)
            elseif ipp_problem.objective == "D-IPP"
                # NOTE: this uses post_gp and not gp since we're asking what is the logdet at the prediction points IF we vist the candidate point?
                # we want to minimize logdet(covariance), so want to go to locations that have greater -logdet(covariance
                rewards[i] = -logdet(cov(post_gp(Ω)))

            elseif ipp_problem.objective == "expected_improvement"
                # NOTE: this uses gp and not post_gp since we're asking what is the EI at the candidate point?
                # we want to visit where EI is the highest
                query_candidate_point = [ipp_problem.Graph.Theta[i, :]]
                y_min = minimum(y_hist)
    
                σ = sqrt.(var(gp(query_candidate_point)))[1]
                μ = mean(gp(query_candidate_point))[1]
                EI = expected_improvement(y_min, μ, σ)
                rewards[i] = EI
            elseif ipp_problem.objective == "lower_confidence_bound"
                # NOTE: this uses gp and not post_gp since we're asking what is the lcb at the candidate point?
                # want to go where μ - α*σ is lowest, so we want to go to locations that have greater (less negative) -(μ - α*σ)
                query_candidate_point = [ipp_problem.Graph.Theta[i, :]]
         
                σ = sqrt.(var(gp(query_candidate_point)))
                μ = mean(gp(query_candidate_point))
                α = 1.0
                rewards[i] = -(μ - α*σ)[1]
            elseif ipp_problem.objective == "catnipp"
                # NOTE: this uses post_gp and not gp since we're asking what is the variance at the high interest nodes IF we vist the candidate point?
                # we want to minimize variance, so we want to go to locations that have greater (less negative) -sum(variance)
                # notice that we use y = mean(gp) to predict the measurement value at the candidate point and DON'T use the ground truth map
                
                if isempty(high_interest_nodes)
                    variances = var(post_gp(Ω))
                    rewards[i] = -sum(variances)
                else
                    variances = var(post_gp(high_interest_nodes))
                    rewards[i] = -sum(variances)
                end
            end
        end
    end

    return unique(reachable_nodes), rewards
end

function aspo_catnipp_solve_dp_orienteering(graph_rewards::Vector, G::Vector{Vector{Int64}}, start::Int, goal::Int, max_steps::Int)
    """
    Solves the orienteering problem on a general graph using dynamic programming.
    Builds a table of size n x max_steps and fills it in using the Bellman equation.
    Returns the optimal path and its value.
    """

    n = length(graph_rewards)

    # DP table, dimensions: n x max_steps, initialized with -infinity
    dp = fill(-Inf, n, max_steps + 1)
    # Path table to reconstruct the path
    path = Array{Union{Nothing, Int}}(nothing, n, max_steps + 1)

    # Base case: at the goal position with 0 steps left
    dp[goal, 1] = graph_rewards[goal]

    # Fill the DP table
    for step in 1:max_steps
        for node in 1:n
            # Check all neighbors of the current node
            for neighbor in G[node]
                new_reward = graph_rewards[node] + dp[neighbor, step]
                if new_reward > dp[node, step + 1]
                    dp[node, step + 1] = new_reward
                    path[node, step + 1] = neighbor
                end
            end
        end
    end

    # Reconstruct the path from start to goal
    current_node = start
    steps_left = max_steps
    optimal_path = [current_node]

    while steps_left > 0 && path[current_node, steps_left + 1] !== nothing
        current_node = path[current_node, steps_left + 1]
        push!(optimal_path, current_node)
        steps_left -= 1
    end

    return dp[start, max_steps + 1], optimal_path
end

function aspo_catnipp_action(ipp_problem::IPP, method::ASPOCAtNIPPComparison, gp::AbstractGPs.PosteriorGP, executed_path::Vector{Int64}, y_hist::Vector{Float64})
    n = ipp_problem.n
    n_sqrt = isqrt(n)
    pos = executed_path[end]

    val, t = @timed aspo_catnipp_estimate_rewards(ipp_problem, gp, executed_path, y_hist)
    reachable_nodes, rewards = val
    # println("Estimate Rewards Time: $(t)")  

    budget_remaining = ipp_problem.B - path_distance(ipp_problem, executed_path)

    # find the nearest even number of steps remaining
    if ipp_problem.B % 2 == 0
        # even budget
        steps_remaining = round(Int, budget_remaining*(n_sqrt-1)/ipp_problem.Graph.edge_length)
    else
        # odd budget
        steps_remaining = round(Int, ipp_problem.B*(n_sqrt-1)/ipp_problem.Graph.edge_length - length(executed_path))
    end

    if steps_remaining < 0
        steps_remaining = 0
    end

    val, t = @timed aspo_catnipp_solve_dp_orienteering(rewards, ipp_problem.Graph.G, pos, ipp_problem.Graph.goal, steps_remaining)
    path_value, planned_path = val

    # println("Orienteering Solution Time: $(t)")

    if path_value == -Inf
        println("No solution found")
        planned_path = Vector{Int64}()#shortest_path(ipp_problem.Graph.all_pairs_shortest_paths, pos, n)
    end

    return planned_path
end

function aspo_catnipp_initialize_gp(ipp_problem::IPP, gp)
    path = [ipp_problem.Graph.start]

    x = ipp_problem.Graph.Theta[path, :]
    X = [x[i, :] for i in 1:size(x, 1)]
    # y = ipp_problem.Graph.true_map[path] + [rand(ipp_problem.rng, Normal(0, ipp_problem.MeasurementModel.σ))]
    y = [InformativePathPlanning.get_measurement(ipp_problem.Graph.Theta[path, :], ipp_problem.Graph.true_map)] + [rand(ipp_problem.rng, Normal(0, ipp_problem.MeasurementModel.σ))]
    y_hist = y
    gp = AbstractGPs.posterior(gp(X, ipp_problem.MeasurementModel.σ^2), y)
    return gp, y_hist
end

function aspo_catnipp_update_gp(ipp_problem::IPP, gp::AbstractGPs.PosteriorGP, y_hist, planned_path)
    x = ipp_problem.Graph.Theta[planned_path, :]
    X = [x[i, :] for i in 1:size(x, 1)]
    
    # y = ipp_problem.Graph.true_map[planned_path] + rand(ipp_problem.rng, MvNormal(zeros(length(planned_path)), Diagonal(ones(length(planned_path)).*ipp_problem.MeasurementModel.σ^2)) )
    # y = ipp_problem.Graph.true_map[planned_path] + [rand(ipp_problem.rng, Normal(0, ipp_problem.MeasurementModel.σ)) for i in 1:length(planned_path)]
    y = InformativePathPlanning.get_measurement_history(planned_path, ipp_problem.Graph.true_map, ipp_problem.Graph.Theta) + [rand(ipp_problem.rng, Normal(0, ipp_problem.MeasurementModel.σ)) for i in 1:length(planned_path)]
    gp = AbstractGPs.posterior(gp(X, ipp_problem.MeasurementModel.σ^2), y)
    y_hist = vcat(y_hist, y...)

    return gp, y_hist
end

function solve(ipp_problem::IPP, method::ASPOCAtNIPPComparison)
    """ 
    Takes in IPP problem definition and returns the path and objective value
    using the solution method specified by method.
    """

    path = Vector{Int64}([ipp_problem.Graph.start])
    gp, y_hist = aspo_catnipp_initialize_gp(ipp_problem, AbstractGPs.GP(with_lengthscale(MaternKernel(ν=1.5), ipp_problem.MeasurementModel.L)))
    time_left = ipp_problem.solution_time
    prev_planned_path = shortest_path(ipp_problem.Graph.all_pairs_shortest_paths, path[end], ipp_problem.Graph.goal)

    while path[end] != ipp_problem.Graph.goal && time_left > 0
        planned_path, planning_time = @timed aspo_catnipp_action(ipp_problem, method, gp, path, y_hist)
        time_left -= planning_time

        if planned_path == Vector{Int64}()
            # if no solution was found then use the previous solution
            planned_path = prev_planned_path
        end

        if length(planned_path[(2+ipp_problem.replan_rate):end]) <= ipp_problem.replan_rate
            # this is our last path since we're within the replan rate of the goal
            push!(path, planned_path[2:end]...)
            gp, y_hist = aspo_catnipp_update_gp(ipp_problem, gp, y_hist, planned_path[2:end])
            prev_planned_path = shortest_path(ipp_problem.Graph.all_pairs_shortest_paths, planned_path[end], ipp_problem.Graph.goal)
            # break
        else
            push!(path, planned_path[2:(2+ipp_problem.replan_rate-1)]...)
            gp, y_hist = aspo_catnipp_update_gp(ipp_problem, gp, y_hist, planned_path[2:(2+ipp_problem.replan_rate-1)])
            prev_planned_path = planned_path[(2+ipp_problem.replan_rate-1):end]
        end    
    end

    if path[end] != ipp_problem.Graph.goal
        sp_to_goal = shortest_path(ipp_problem.Graph.all_pairs_shortest_paths, path[end], ipp_problem.Graph.goal)[2:end]
        push!(path, sp_to_goal...)
        gp, y_hist = aspo_catnipp_update_gp(ipp_problem, gp, y_hist, sp_to_goal)
    end

    if ipp_problem.objective == "expected_improvement"
        return path, adaptive_objective(ipp_problem, path, y_hist), y_hist
    elseif ipp_problem.objective == "catnipp"
        return path, catnipp_objective(ipp_problem, path)
    else
        return path, objective(ipp_problem, path)
    end
end