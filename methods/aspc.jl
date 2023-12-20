function estimate_rewards(ipp_problem::IPP, gp::AbstractGPs.PosteriorGP, path::Vector{Int64}, y_hist::Vector{Float64})
    """
    Estimates the rewards for all reachable nodes in the graph given the current path and the GP. 
    This assumes we can teleport to any node in the graph (i.e. we can visit any node in the graph without having to visit all nodes in the path to get there).
    We use the GP to estimate the value of each node in the graph depending on the problem objective. 
    """

    n = ipp_problem.n
    n_sqrt = isqrt(n)
    reachable_nodes = Vector{Int64}()
    rewards = -1e6 .* ones(n_sqrt, n_sqrt)
    Ω = [ipp_problem.Graph.Omega[i, :] for i in 1:size(ipp_problem.Graph.Omega, 1)]

    for i in 1:n
        if i ∈ path
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
            post_gp = AbstractGPs.posterior(gp(X, ipp_problem.MeasurementModel.σ), y)
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
            end
        end
    end

    return unique(reachable_nodes), rewards
end

function solve_dp_orienteering(grid_rewards::Matrix, start::Int, goal::Int, max_steps)
    """
    Solves the orienteering problem using dynamic programming. Builds a table of grid_dim x grid_dim x max_steps
    and fills it in using the Bellman equation. Returns the optimal path and its value.
    """

    start_pos = (CartesianIndices(grid_rewards)[start][1], CartesianIndices(grid_rewards)[start][2])
    end_pos = (CartesianIndices(grid_rewards)[goal][1], CartesianIndices(grid_rewards)[goal][2])

    grid_dim = size(grid_rewards, 1)

    # DP table, dimensions: grid_dim x grid_dim x max_steps, initialized with -infinity
    dp = fill(-Inf, grid_dim, grid_dim, max_steps + 1)
    # Path table to reconstruct the path
    path = Array{Union{Nothing, Tuple{Int,Int}}}(nothing, grid_dim, grid_dim, max_steps + 1)

    # Base case: at the end position with 0 steps left
    dp[end_pos[1], end_pos[2], 1] = grid_rewards[end_pos[1], end_pos[2]]
    grid_movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Fill the DP table
    for step in 1:max_steps
        for r in 1:grid_dim
            for c in 1:grid_dim
                # Check all four directions (up, down, left, right)
                for (dr, dc) in grid_movements
                    nr, nc = r + dr, c + dc
                    if 1 <= nr <= grid_dim && 1 <= nc <= grid_dim
                        new_reward = grid_rewards[r, c] + dp[nr, nc, step]
                        if new_reward > dp[r, c, step + 1]
                            dp[r, c, step + 1] = new_reward
                            path[r, c, step + 1] = (nr, nc)
                        end
                    end
                end
            end
        end
    end

    # Reconstruct the path from start to end
    current_pos = start_pos
    steps_left = max_steps
    optimal_path = [current_pos]

    while steps_left > 0 && path[current_pos[1], current_pos[2], steps_left + 1] !== nothing
        current_pos = path[current_pos[1], current_pos[2], steps_left + 1]
        push!(optimal_path, current_pos)
        steps_left -= 1
    end

    function convert_cartesian_to_linear_path(path, grid_rewards)
        linear_indices = LinearIndices(grid_rewards)
        return [linear_indices[i,j] for (i,j) in path]
    end
    
    linear_path = convert_cartesian_to_linear_path(optimal_path, grid_rewards)
    return dp[start_pos[1], start_pos[2], max_steps + 1], linear_path
end

function action(ipp_problem::IPP, method::ASPC, gp::AbstractGPs.PosteriorGP, executed_path::Vector{Int64}, y_hist::Vector{Float64})
    n = ipp_problem.n
    n_sqrt = isqrt(n)
    pos = executed_path[end]
    reachable_nodes, rewards = estimate_rewards(ipp_problem, gp, executed_path, y_hist)

    budget_remaining = ipp_problem.B - path_distance(ipp_problem, executed_path)
    steps_remaining = round(Int, budget_remaining*(n_sqrt-1)/ipp_problem.Graph.edge_length)

    # find the nearest even number of steps remaining
    if steps_remaining % 2 != 0
        steps_remaining += 1
    end

    val, t = @timed solve_dp_orienteering(rewards, pos, ipp_problem.Graph.goal, steps_remaining)
    path_value, planned_path = val

    println("Orienteering Solution Time: $(t)")

    if path_value == -Inf
        println("No solution found")
        planned_path = shortest_path(ipp_problem.Graph.all_pairs_shortest_paths, pos, n)
    end

    return planned_path
end

function solve(ipp_problem::IPP, method::ASPC)
    """ 
    Takes in IPP problem definition and returns the path and objective value
    using the solution method specified by method.
    """

    path = Vector{Int64}([ipp_problem.Graph.start])
    gp, y_hist = initialize_gp(ipp_problem)
    time_left = ipp_problem.solution_time

    while path[end] != ipp_problem.Graph.goal && time_left > 0
        planned_path, planning_time = @timed action(ipp_problem, method, gp, path, y_hist)
        time_left -= planning_time

        if length(planned_path[(2+ipp_problem.replan_rate):end]) <= ipp_problem.replan_rate
            push!(path, planned_path[(2+ipp_problem.replan_rate):end]...)
            gp, y_hist = update_gp(ipp_problem, gp, y_hist, planned_path[(2+ipp_problem.replan_rate):end])
            break
        else
            push!(path, planned_path[2:(2+ipp_problem.replan_rate-1)]...)
            gp, y_hist = update_gp(ipp_problem, gp, y_hist, planned_path[2:(2+ipp_problem.replan_rate-1)])
        end    
    end

    return path, objective(ipp_problem, path, y_hist)
end