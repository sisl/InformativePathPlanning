function Plots.plot(ipp_problem::IPP, path::Vector{Int}, objVal::Float64, runtime::Float64, figure_path::String="figures/1.pdf")
    """
    Standard IPP plotting 
    """
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

function Plots.plot(mmipp::MultimodalIPP, path::Vector{Int}, drills::Vector{Int}, objVal::Float64, runtime::Float64, figure_path::String="figures/1.pdf")
    """
    Plots Multimodal IPP problem with drill locations 
    """
    ipp_problem = mmipp.ipp_problem
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
    # scatter the drill locations
    scatter!([Theta[drills[i], 1] for i in 1:length(drills)], [Theta[drills[i], 2] for i in 1:length(drills)], label="drill loc", c=:yellow)
    savefig(figure_path)
end

function Plots.plot(mmipp::MultimodalIPP, path::Vector{Int}, mcts_drills::Vector{Int}, cvx_drills::Vector{Int}, objVal::Float64, runtime::Float64, figure_path::String="figures/1.pdf")
    """
    Plots Multimodal IPP problem with both MCTS and CVX drill locations
    """
    ipp_problem = mmipp.ipp_problem
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
    # scatter the drill locations
    scatter!([Theta[mcts_drills[i], 1] for i in 1:length(mcts_drills)], [Theta[mcts_drills[i], 2] for i in 1:length(mcts_drills)], label="mcts drills", c=:yellow)
    scatter!([Theta[cvx_drills[i], 1] for i in 1:length(cvx_drills)], [Theta[cvx_drills[i], 2] for i in 1:length(cvx_drills)], label="cvx drills", c=:red)
    savefig(figure_path)
end

function Plots.plot(mipp::MultiagentIPP, paths_hist, planned_paths_hist, gp_hist, centers, radii)
    """
    Plots Multiagent IPP gif
    """
    planning_steps = length(gp_hist)
    n = mipp.ipp_problem.n
    n_sqrt = isqrt(n)

    Theta = mipp.ipp_problem.Graph.Theta
    Omega = mipp.ipp_problem.Graph.Omega
    L = mipp.ipp_problem.MeasurementModel.L
    plot_scale = range(0, mipp.ipp_problem.Graph.edge_length, length=100) #1:0.1:10
    X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    plot_Omega = Matrix(hcat(X_plot...)')
    Ω = [plot_Omega[i, :] for i in 1:size(plot_Omega, 1)]
    agent_colors = [:blue, :red, :white]


    @show planning_steps
    @show length(planned_paths_hist)
    anim = @animate for i in 1:planning_steps
        @show i 
        # heatmap of GP variance
        post_gp = gp_hist[i]
        fxs = var(post_gp(Ω))
        heatmap(collect(plot_scale), collect(plot_scale), reshape(fxs, plot_size...), c = cgrad(:inferno, rev = false), xaxis=false, yaxis=false, grid=false, clim=(0,1))

        G = mipp.ipp_problem.Graph.G
        nodes_in_new_G = [G[i] for i in 1:length(G)]
        nodes_in_new_G = unique(vcat(nodes_in_new_G...))
        # scatter!([Theta[i, 2] for i in nodes_in_new_G], [Theta[i, 1] for i in nodes_in_new_G])

        # plot obstacles using centers and radii
        # for j in 1:size(centers)[1]
        #     x = centers[j, 1] ./ n_sqrt
        #     y = centers[j, 2] ./ n_sqrt
        #     r = radii[j] ./ n_sqrt
        #     plot!((x .+ r*cos.(range(0, 2*pi, length=100))), (y .+ r*sin.(range(0, 2*pi, length=100))), color=:black, linewidth=3, label="Obstacle")
        # end
        nodes_in_obstacles = [i for i in 1:n if i ∉ nodes_in_new_G]
        scatter!([Theta[i, 2] for i in nodes_in_obstacles], [Theta[i, 1] for i in nodes_in_obstacles], color=:black, label="Obstacle", markersize=6)
        scatter!(Omega[:, 2], Omega[:, 1], color=:white, label="Prediction Location", markersize=3)

        for j in 1:mipp.M
            if i > length(paths_hist[j])
                path = paths_hist[j][end]
                plot!(Theta[path, 2], Theta[path, 1], color=agent_colors[j], linewidth=3, label="Agent $(j)")
                continue
            end
            # plot executed path so far for agent i 
            path = paths_hist[j][i]
            plot!(Theta[path, 2], Theta[path, 1], color=agent_colors[j], linewidth=3, label="Agent $(j)")

            # plot planned path for agent i
            planned_path = planned_paths_hist[j][i]
            plot!(Theta[planned_path, 2], Theta[planned_path, 1], color=agent_colors[j], linewidth=3, linestyle=:dash, label="")
        end
        # plot the legend so it is to the side of the plot but not on top of the plot
        plot!(legend=:outerleft, size=(1000, 500))
    end
    Plots.gif(anim,  "figures/multiagent.gif", fps = 5)
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
