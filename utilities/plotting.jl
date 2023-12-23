function Plots.plot(ipp_problem::IPP, path::Vector{Int}, objVal::Float64, runtime::Float64, figure_path::String="figures/1.pdf")
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
    savefig(figure_path)
end

function Plots.plot(mipp::MultiagentIPP, paths_hist, planned_paths_hist, gp_hist)
    planning_steps = length(gp_hist)

    Theta = mipp.ipp_problem.Graph.Theta
    L = mipp.ipp_problem.MeasurementModel.L
    plot_scale = range(0, mipp.ipp_problem.edge_length, length=100) #1:0.1:10
    X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    plot_Omega = Matrix(hcat(X_plot...)')
    Ω = [plot_Omega[i, :] for i in 1:size(plot_Omega, 1)]
    agent_colors = [:blue, :red, :white]

    anim = @animate for i in planning_steps
        # heatmap of GP variance
        gp = gp_hist[i]
        fxs = var(post_gp(Ω))
        heatmap(collect(plot_scale), collect(plot_scale), reshape(fxs, plot_size...), c = cgrad(:inferno, rev = false), xaxis=false, yaxis=false, legend=false, grid=false, clim=(0,1))
        
        for j in 1:mipp.M
            # plot executed path so far for agent i 
            path = paths_hist[j][i]
            plot!(Theta[path, 2], Theta[path, 1], color=agent_colors[j], linewidth=3, label="Agent $(j)")

            # plot planned path for agent i
            planned_path = planned_paths_hist[j][i]
            plot!(Theta[planned_path, 2], Theta[planned_path, 1], color=:yellow, linewidth=3, linestyle=:dash)
        end
    end

    gif(anim, "figures/multiagent.gif", fps = 1)
end

    

#  # plot the heatmap of optimal u and the path over it
#  n_sqrt = isqrt(n)
#  heatmap(collect((range(0, ipp_problem.Graph.edge_length, n_sqrt))), collect((range(0, ipp_problem.Graph.edge_length, n_sqrt))), reshape(optimal_u, (n_sqrt, n_sqrt)), title="Optimal u", xlabel="x", ylabel="y", colorbar=false, size=(500, 500))
#  plot!(ipp_problem.Graph.Theta[path, 2], ipp_problem.Graph.Theta[path, 1], color=:orchid1, linewidth=3, label="Path")
#  savefig("figures/BIPP_$(n)n_$(ipp_problem.objective)_$(rand(1:100)).png")

#  flow_z = reshape([sum([optimal_z[(pos, j)] for j in G[pos]] + [optimal_z[(j, pos)] for j in G[pos]]) for pos in 1:n], (n_sqrt, n_sqrt))
#  heatmap(collect((range(0, ipp_problem.Graph.edge_length, n_sqrt))), collect((range(0, ipp_problem.Graph.edge_length, n_sqrt))), flow_z, title="Optimal z", xlabel="x", ylabel="y", colorbar=false, size=(500, 500))
#  plot!(ipp_problem.Graph.Theta[path, 2], ipp_problem.Graph.Theta[path, 1], color=:orchid1, linewidth=3, label="Path")
#  savefig("figures/z_BIPP_$(n)n_$(ipp_problem.objective)_$(rand(1:100)).png")

#  heatmap(A)
#  savefig("figures/A_$(n)n_$(ipp_problem.objective)_$(rand(1:100)).png")
