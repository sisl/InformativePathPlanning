function greedy_reward(ipp_problem::IPP, gp::AbstractGPs.PosteriorGP, candidate_point::Int64, Ω, y_hist)
    Theta = ipp_problem.Graph.Theta
    objective = ipp_problem.objective

    x = Theta[[candidate_point], :]
    X = [x[i, :] for i in 1:size(x, 1)]

    # use gp mean to predict value of candidate point
    y = mean(gp([Theta[candidate_point, :]]))[1]

    y_min = minimum(y_hist)
    post_gp = AbstractGPs.posterior(gp(X, ipp_problem.MeasurementModel.σ), [y])
    
    if objective == "A-IPP"
        # NOTE: this uses post_gp and not gp since we're asking what is the variance at the prediction points IF we vist the candidate point?
        variances = var(post_gp(Ω))
        return sum(variances)
    elseif objective == "D-IPP"
        # NOTE: this uses post_gp and not gp since we're asking what is the logdet at the prediction points IF we vist the candidate point?
        return logdet(cov(post_gp(Ω)))
    elseif objective == "expected_improvement"
        # NOTE: this uses gp and not post_gp since we're asking what is the EI at the candidate point?
        # want to go where EI is highest (where -EI is lowest)
        query_candidate_point = [Theta[candidate_point, :]]

        σ = sqrt.(var(gp(query_candidate_point)))[1]
        μ = mean(gp(query_candidate_point))[1]
        EI = expected_improvement(y_min, μ, σ)
        return -EI
    elseif objective == "lower_confidence_bound"
        # NOTE: this uses gp and not post_gp since we're asking what is the lcb at the candidate point?
        # want to go where μ - α*σ is lowest
        query_candidate_point = [Theta[candidate_point, :]]
         
        σ = sqrt.(var(gp(query_candidate_point)))
        μ = mean(gp(query_candidate_point))
        α = 1.0
        return (μ - α*σ)[1]
    end
end

function action(ipp_problem::IPP, method::Greedy, gp::AbstractGPs.PosteriorGP, executed_path::Vector{Int64}, Ω, y_hist)
    n = ipp_problem.n
    n_sqrt = isqrt(n)
    pos = executed_path[end]

    positions = ipp_problem.Graph.G[pos]
    
    best_fxs = Inf
    best_candidate_point = -1

    for candidate_point in positions
        path_to_node = vcat(vcat(executed_path[1:end-1], shortest_path(ipp_problem.Graph.all_pairs_shortest_paths, executed_path[end], candidate_point)), shortest_path(ipp_problem.Graph.all_pairs_shortest_paths, candidate_point, ipp_problem.Graph.goal)[2:end])

        # check if node is reachable
        # @show path_distance(ipp_problem, path_to_node)
        if round(path_distance(ipp_problem, path_to_node), digits=5) <= ipp_problem.B
            can_fxs = greedy_reward(ipp_problem, gp, candidate_point, Ω, y_hist)

            if can_fxs < best_fxs
                best_fxs, best_candidate_point = can_fxs, candidate_point
            end
        end
    end
    
    next_pos = best_candidate_point
    return next_pos
end

function solve(ipp_problem::IPP, method::Greedy)
    """ 
    Takes in IPP problem definition and returns the path and objective value
    using the solution method specified by method.
    """

    path = Vector{Int64}([ipp_problem.Graph.start])
    gp, y_hist = initialize_gp(ipp_problem)
    time_left = ipp_problem.solution_time

    Ω = [ipp_problem.Graph.Omega[i, :] for i in 1:size(ipp_problem.Graph.Omega, 1)] 
    while path[end] != ipp_problem.Graph.goal && time_left > 0
        next_pos = action(ipp_problem, method, gp, path, Ω, y_hist)
        push!(path, next_pos)
        gp, y_hist = update_gp(ipp_problem, gp, y_hist, [next_pos])
    end

    if ipp_problem.objective == "expected_improvement"
        return path, objective(ipp_problem, path, y_hist), y_hist
    else
        return path, objective(ipp_problem, path, y_hist)
    end
end