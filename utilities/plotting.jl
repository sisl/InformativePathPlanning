function Plots.plot(ipp_problem::IPP, path::Vector{Int}, objVal::Float64, runtime::Float64, figures_path::String="figures/", run_idx::Int=1)
    objective = ipp_problem.objective
    B = ipp_problem.B
    n = ipp_problem.n
    m = ipp_problem.m
    true_map = ipp_problem.Graph.true_map
    edge_length = ipp_problem.Graph.edge_length
    Theta = ipp_problem.Graph.Theta
    Omega = ipp_problem.Graph.Omega

    if objective == "expected_improvement" || objective == "lower_confidence_bound"
        plot_scale = collect(range(0,edge_length, size(true_map, 1)))

        @show size(true_map)
        @show isqrt(size(true_map, 1))
        @show length(plot_scale)
        heatmap(plot_scale, plot_scale, true_map')
    else
        scatter(Theta[:, 1], Theta[:, 2], label="obs loc")
        scatter!(Omega[:, 1], Omega[:, 2], label="pred loc")
    end
    pLength = path_distance(ipp_problem, path)
    plot!([Theta[path[i], 1] for i in 1:length(path)], [Theta[path[i], 2] for i in 1:length(path)], label="path", lw=3, c=:black, title="N = $(n), M = $(m), Obj=$(round(objVal, digits=2)), B=$(B), d=$(round(pLength, digits=2)), t=$(round(runtime, digits=2))")
    savefig(figures_path * "$(run_idx).pdf")
end