function path_distance(ipp_problem::IPP, path::Vector{Int64})
    d = ipp_problem.Graph.distances
    if length(path) == 1
        return 0.0
    end
    return sum(d[path[i], path[i+1]] for i in 1:(length(path)-1))
end

function shortest_path(all_pairs_shortest_paths::Graphs.FloydWarshallState{Float64, Int64}, source::Int, target::Int)
    path = [target]
    while target != source
        target = all_pairs_shortest_paths.parents[source, target]
        pushfirst!(path, target)
    end
    return path
end

function kernel(x::Vector{Vector{Float64}}, y::Vector{Vector{Float64}}, L, σ_0=1)
    K = kernelmatrix(with_lengthscale(SqExponentialKernel(), L), x, y)
    return K
end

function kernel(x::Matrix{Float64}, y::Matrix{Float64}, L, σ_0=1)
    K = kernelmatrix(with_lengthscale(SqExponentialKernel(), L), x', y')
    return K
end

function initialize_gp(ipp_problem::IPP)
    path = [ipp_problem.Graph.start]
    gp = AbstractGPs.GP(with_lengthscale(SqExponentialKernel(), ipp_problem.MeasurementModel.L))

    x = ipp_problem.Graph.Theta[path, :]
    X = [x[i, :] for i in 1:size(x, 1)]
    y = ipp_problem.Graph.true_map[path] + [rand(ipp_problem.rng, Normal(0, ipp_problem.MeasurementModel.σ))]
    y_hist = y
    gp = AbstractGPs.posterior(gp(X, ipp_problem.MeasurementModel.σ^2), y)
    return gp, y_hist
end

function update_gp(ipp_problem::IPP, gp::AbstractGPs.PosteriorGP, y_hist, planned_path)
    x = ipp_problem.Graph.Theta[planned_path, :]
    X = [x[i, :] for i in 1:size(x, 1)]
    
    # y = ipp_problem.Graph.true_map[planned_path] + rand(ipp_problem.rng, MvNormal(zeros(length(planned_path)), Diagonal(ones(length(planned_path)).*ipp_problem.MeasurementModel.σ^2)) )
    y = ipp_problem.Graph.true_map[planned_path] + [rand(ipp_problem.rng, Normal(0, ipp_problem.MeasurementModel.σ)) for i in 1:length(planned_path)]
    gp = AbstractGPs.posterior(gp(X, ipp_problem.MeasurementModel.σ^2), y)
    y_hist = vcat(y_hist, y...)

    return gp, y_hist
end

function objective(mmipp::MultimodalIPP, path::Vector{Int64}, y_hist::Vector{Float64}, drills::Vector{Int64})
    ipp_problem = mmipp.ipp_problem
    σ_max = mmipp.σ_max
    σ_min = mmipp.σ_min

    gp = AbstractGPs.GP(with_lengthscale(SqExponentialKernel(), ipp_problem.MeasurementModel.L))
    Ω = [ipp_problem.Graph.Omega[i, :] for i in 1:size(ipp_problem.Graph.Omega, 1)]

    ν = σ_max^2 .* ones(1:length(path))
    drill_path_idx = [findfirst(x->x==i, path) for i in drills]
    ν[drill_path_idx] .= σ_min

    if length(y_hist) != length(path)
        @show length(y_hist)
        @show length(path)
        @error("length(y_hist) != length(path)")
    else
        x = ipp_problem.Graph.Theta[path, :]
        X = [x[i, :] for i in 1:size(x, 1)]    
        y = y_hist
        post_gp = AbstractGPs.posterior(gp(X, ν), y)
    end

    if ipp_problem.objective == "A-IPP"
        variances = var(post_gp(Ω))
        return sum(variances)
    elseif ipp_problem.objective == "D-IPP"
        return logdet(cov(post_gp(Ω)))
    elseif ipp_problem.objective == "expected_improvement"
        # want to go where EI is highest (where -EI is lowest)
        # Ω = [ipp_problem.Graph.Theta[i, :] for i in 1:size(ipp_problem.Graph.Theta, 1)]
        y_min = minimum(y_hist)

        σ = sqrt.(var(post_gp(Ω)))
        μ = mean(post_gp(Ω))
        EI = [expected_improvement(y_min, μ[i], σ[i]) for i in 1:length(μ)]
        return sum(EI)
    elseif ipp_problem.objective == "lower_confidence_bound"
        # want to go where μ - α*σ is lowest
        # Ω = [ipp_problem.Graph.Theta[i, :] for i in 1:size(ipp_problem.Graph.Theta, 1)]

        σ = sqrt.(var(post_gp(Ω)))
        μ = mean(post_gp(Ω))
        α = 1.0
        return sum(μ - α*σ)
    end
end

function adaptive_objective(ipp_problem::IPP, path::Vector{Int64}, y_hist::Vector{Float64})
    """
    Compute the objective function value for a given path and history of measurements. To be used 
    for adaptive objectives where the objective depends on the measurements taken.
    """
    if ipp_problem.objective == "A-IPP" || ipp_problem.objective == "D-IPP"
        @error("wrong objective function")
    end

    gp = AbstractGPs.GP(with_lengthscale(SqExponentialKernel(), ipp_problem.MeasurementModel.L))
    Ω = [ipp_problem.Graph.Omega[i, :] for i in 1:size(ipp_problem.Graph.Omega, 1)]

    ν = ipp_problem.MeasurementModel.σ^2 .* ones(1:length(path))

    if length(y_hist) != length(path)
        @show length(y_hist)
        @show length(path)
        @error("length(y_hist) != length(path)")
    else
        x = ipp_problem.Graph.Theta[path, :]
        X = [x[i, :] for i in 1:size(x, 1)]    
        y = y_hist
        post_gp = AbstractGPs.posterior(gp(X, ν), y)
    end

    if ipp_problem.objective == "expected_improvement"
        # want to go where EI is highest (where -EI is lowest)
        # Ω = [ipp_problem.Graph.Theta[i, :] for i in 1:size(ipp_problem.Graph.Theta, 1)]
        y_min = minimum(y_hist)

        σ = sqrt.(var(post_gp(Ω)))
        μ = mean(post_gp(Ω))
        EI = [expected_improvement(y_min, μ[i], σ[i]) for i in 1:length(μ)]
        return sum(EI)
    elseif ipp_problem.objective == "lower_confidence_bound"
        # want to go where μ - α*σ is lowest
        # Ω = [ipp_problem.Graph.Theta[i, :] for i in 1:size(ipp_problem.Graph.Theta, 1)]

        σ = sqrt.(var(post_gp(Ω)))
        μ = mean(post_gp(Ω))
        α = 1.0
        return sum(μ - α*σ)
    end
end

function objective(ipp_problem::IPP, path::Vector{Int64}) 
    """
    Standard objective used for A and D-IPP
    """
    A = ipp_problem.MeasurementModel.A
    Σₓ⁻¹ = ipp_problem.MeasurementModel.Σₓ⁻¹
    σ = ipp_problem.MeasurementModel.σ
    n = ipp_problem.n

    ZA_mat_actual = σ^(-2)*sum((i in path)*A[i, :]*A[i, :]' for i in 1:n)
    Σ_est_inv_actual = Symmetric(ZA_mat_actual + Σₓ⁻¹)
    Σ_est_actual = Symmetric(inv(Σ_est_inv_actual))

    if ipp_problem.objective == "A-IPP"
        return tr(Σ_est_actual)
    elseif ipp_problem.objective == "D-IPP"
        return logdet(Σ_est_actual)
    else
        @error("objective function not implemented")
    end
end

prob_of_improvement(y_min, μ, σ) = cdf(Normal(μ, σ), y_min) 

function expected_improvement(y_min, μ, σ)
    p_imp = prob_of_improvement(y_min, μ, σ)
    p_ymin = pdf(Normal(μ, σ), y_min)
    return (y_min - μ)*p_imp + (σ^2)*p_ymin
end

function compute_adaptive_obj_hist(ipp_problem::IPP, y_hist, path)
    obj_hist = zeros(length(y_hist))
    for i in 1:length(y_hist)
        obj_hist[i] = adaptive_objective(ipp_problem, path[1:i], y_hist[1:i])
    end
    return obj_hist
end