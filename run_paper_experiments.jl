include("IPP.jl")
using JLD2
using Plots
using Statistics
using Measures 
using ColorSchemes

@with_kw struct SimulationData
    sim_number::Int
    run_type::SolutionMethod
    n::Int
    m::Int
    B::Int
    L::Float64
    replan_rate::Int
    timeout::Float64
    σ_min::Float64
    σ_max::Float64
    objVal::Float64
    y_hist::Vector{Float64}
    lower_bound::Float64
    upper_bound::Float64
    path::Vector{Int}
    drills::Vector{Int}
    Omega::Matrix{Float64}
    runtime::Float64
end

scheme = ColorSchemes.tab10
aspc_color = scheme[1]
exact_color = scheme[2]
mcts_color = scheme[3]
random_color = scheme[4]
greedy_color = scheme[5]
mip_color = scheme[6]
trΣ⁻¹_color = scheme[7]
relaxed_color = scheme[8]
theme(:default)
default(titlefont=font(18, "Computer Modern"))
default(guidefont=font(16, "Computer Modern"))
default(tickfont=font(14, "Computer Modern"))
default(legendfont=font(12, "Computer Modern"))

function std_err(data, n)
    @show length(data)
    return std(data, dims=1)' / sqrt(n)
end

#####################################################################################
# Figure 1
#####################################################################################
function figure_1(load_data=false, data_path="data/")
    """
    Runtime and A-IPP objective as a function of the graph size. We compare 
    the MIP formulation of Dutta et al. with our exact MIP formulation and
    the B-IPP objective. 
    """
    grid_nodes = collect(62:-6:2).^2 # 65 dutta failed
    num_sims = 25
    edge_length = 1 # 100
    L = 0.01*edge_length # length scale 
    σ = 1.0
    objective = "A-IPP"
    solution_time = 120.0
    rng = MersenneTwister(12345)
    m = 20
    B = 4*edge_length

    if load_data
        data = JLD2.load(data_path * "figure_1.jld2", "data")
    else
        data = []
        methods = [DuttaMIP(), Exact(), trΣ⁻¹()]

        p = Progress(length(grid_nodes)*num_sims*length(methods))

        for (n_idx, n) in enumerate(grid_nodes)
            start = 1
            goal = n
            replan_rate = round(Int, 0.05 * B/edge_length * sqrt(n)) #round(Int, 0.1 * B/edge_length * sqrt(n))

            # true map doesn't matter for non-adaptive objectives
            true_map = rand(rng, isqrt(n), isqrt(n))

            # Generate a grid graph
            Graph = build_graph(rng, data_path, n, m, edge_length, start, goal, objective)

            for (method_idx, method) in enumerate(methods)
                for i in 1:num_sims
                    shared_idx = (n_idx - 1) * length(methods) * num_sims + (method_idx - 1) * num_sims + i
                    # shared_idx = (n_idx-1)*length(methods)*num_sims + i
                    println("##########################################################################################")
                    println( string(shared_idx) * "/" * string(length(grid_nodes)*num_sims*length(methods)) * " grid_nodes " * string(n) * " run_type " * string(method))
                    println("##########################################################################################")

                    # Here we have to change only the Omega's 
                    omega_x = rand(rng, m)*edge_length
                    omega_y = rand(rng, m)*edge_length
                    Omega = hcat(omega_x, omega_y)
                    Graph = IPPGraph(Graph.G, Graph.start, Graph.goal, Graph.Theta, Omega, Graph.all_pairs_shortest_paths, Graph.distances, Graph.true_map, Graph.edge_length)

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

                    # Solve the IPP problem
                    val, t = @timed solve(ipp_problem, method)
                    path, objective_value = val
                    new_data = SimulationData(sim_number=shared_idx, run_type=method, n=n, m=m, B=B, L=L, replan_rate=replan_rate, timeout=solution_time, σ_min=1e-5, σ_max=σ, objVal=objective_value, y_hist=Vector{Float64}(), lower_bound=0.0, upper_bound=0.0, path=path, drills=Vector{Int}(), Omega=ipp_problem.Graph.Omega, runtime=t)
                    push!(data, new_data)

                    @show objective_value
                    @show t

                    # Plot the IPP problem
                    plot(ipp_problem, path, objective_value, t, "figures/paper/figure_1/runs/$(typeof(method))_$(n)n_$(objective)_$(i).pdf")
                    next!(p)
                    sleep(0.1)
                end
            end
        end
        JLD2.save(data_path * "figure_1.jld2", "data", data)
    end
    # Plotting 
    mip_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == DuttaMIP])
    exact_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == Exact])
    trΣ⁻¹_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == trΣ⁻¹])

    mip_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == DuttaMIP], (num_sims, length(mip_xdata))) 
    exact_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == Exact], (num_sims, length(exact_xdata)))
    trΣ⁻¹_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == trΣ⁻¹], (num_sims, length(trΣ⁻¹_xdata)))

    ###################################
    # Runtime vs. Number of Graph Nodes
    plot(color_palette=:tab10)
    plot(mip_xdata, mean(mip_ydata, dims=1)', ribbon = std_err(mip_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "MIP", color=mip_color, yscale=:log10)
    plot!(exact_xdata, mean(exact_ydata, dims=1)', ribbon = std_err(exact_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Exact", color=exact_color, yscale=:log10)
    plot!(trΣ⁻¹_xdata, mean(trΣ⁻¹_ydata, dims=1)', ribbon = std_err(trΣ⁻¹_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "trΣ⁻¹", color=trΣ⁻¹_color, yscale=:log10)
    rt_plot = plot!(mip_xdata, data[1].timeout .* ones(size(mip_xdata)), label = "Timeout", color=:black, linestyle=:dash, linewidth=2, legend=false, yscale=:log10, dpi=500, widen=false, margin=5mm, size=(600,500))
    ###################################
    
    mip_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == DuttaMIP], (num_sims, length(mip_xdata)))
    exact_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == Exact], (num_sims, length(exact_xdata)))
    trΣ⁻¹_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == trΣ⁻¹], (num_sims, length(trΣ⁻¹_xdata))) 
    
    plot(color_palette=:tab10)
    plot(mip_xdata, mean(mip_ydata, dims=1)', ribbon = std_err(mip_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "MIP", color=mip_color)
    plot!(exact_xdata, mean(exact_ydata, dims=1)', ribbon = std_err(exact_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Exact", color=exact_color)
    obj_plot = plot!(trΣ⁻¹_xdata, mean(trΣ⁻¹_ydata, dims=1)', ribbon = std_err(trΣ⁻¹_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "trΣ", color=trΣ⁻¹_color, title="tr(Σ) vs. Graph Size", legend=false, framestyle=:box, widen=false, margin=5mm, size=(600,500))
    plot([rt_plot, obj_plot]..., size=(1200, 500), layout=(1,2), margin=8mm, legend=false, widen=false)
    savefig("figures/paper/figure_1/mip_exact_tr_comparison.pdf")
end


#####################################################################################
# Figure 2
#####################################################################################
function figure_2(load_data=false, data_path="data/")
    """
    Runtime and A-IPP objective as a function of the graph size.
    """
    grid_nodes = collect(123:-12:4).^2
    num_sims = 25
    edge_length = 1
    L = 0.01*edge_length # length scale 
    σ = 1.0
    objective = "A-IPP" 
    solution_time = 120.0
    rng = MersenneTwister(12345)
    m = 20
    B = 4*edge_length

    if load_data
        data = JLD2.load(data_path * "figure_2.jld2", "data")
    else
        data = []
        methods = [Exact(), trΣ⁻¹(), Greedy(), ASPC(), mcts(), random()]

        p = Progress(length(grid_nodes)*num_sims*length(methods))

        for (n_idx, n) in enumerate(grid_nodes)
            start = 1
            goal = n
            replan_rate = round(Int, 0.05 * B/edge_length * sqrt(n))#round(Int, 0.1 * B/edge_length * sqrt(n))

            # true map doesn't matter for non-adaptive objectives
            true_map = rand(rng, isqrt(n), isqrt(n))

            # Generate a grid graph
            Graph = build_graph(rng, data_path, n, m, edge_length, start, goal, objective)

            for (method_idx, method) in enumerate(methods)
                for i in 1:num_sims
                    shared_idx = (n_idx - 1) * length(methods) * num_sims + (method_idx - 1) * num_sims + i
                    # shared_idx = (n_idx-1)*length(methods)*num_sims + i
                    println("##########################################################################################")
                    println( string(shared_idx) * "/" * string(length(grid_nodes)*num_sims*length(methods)) * " grid_nodes " * string(n) * " run_type " * string(method))
                    println("##########################################################################################")

                    # Here we have to change only the Omega's 
                    omega_x = rand(rng, m)*edge_length
                    omega_y = rand(rng, m)*edge_length
                    Omega = hcat(omega_x, omega_y)
                    Graph = IPPGraph(Graph.G, Graph.start, Graph.goal, Graph.Theta, Omega, Graph.all_pairs_shortest_paths, Graph.distances, Graph.true_map, Graph.edge_length)

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

                    # Solve the IPP problem
                    val, t = @timed solve(ipp_problem, method)
                    path, objective_value = val
                    new_data = SimulationData(sim_number=shared_idx, run_type=method, n=n, m=m, B=B, L=L, replan_rate=replan_rate, timeout=solution_time, σ_min=1e-5, σ_max=σ, objVal=objective_value, y_hist=Vector{Float64}(), lower_bound=0.0, upper_bound=0.0, path=path, drills=Vector{Int}(), Omega=ipp_problem.Graph.Omega, runtime=t)
                    push!(data, new_data)

                    @show objective_value
                    @show t

                    # Plot the IPP problem
                    plot(ipp_problem, path, objective_value, t, "figures/paper/figure_2/runs/$(typeof(method))_$(n)n_$(objective)_$(i).pdf")
                    next!(p)
                    sleep(0.1)
                end
            end
        end
        JLD2.save(data_path * "figure_2.jld2", "data", data)
    end
    # Plotting 
    exact_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == Exact])
    trΣ⁻¹_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == trΣ⁻¹])
    greedy_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == Greedy])
    aspc_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == ASPC])
    mcts_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == mcts])
    random_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == random])

    exact_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == Exact], (num_sims, length(exact_xdata)))
    trΣ⁻¹_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == trΣ⁻¹], (num_sims, length(trΣ⁻¹_xdata)))
    greedy_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == Greedy], (num_sims, length(greedy_xdata)))
    aspc_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == ASPC], (num_sims, length(aspc_xdata)))
    mcts_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == mcts], (num_sims, length(mcts_xdata)))
    random_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == random], (num_sims, length(random_xdata)))

    ###################################
    # Runtime vs. Number of Graph Nodes
    plot(color_palette=:tab10)
    plot!(exact_xdata, mean(exact_ydata, dims=1)', ribbon = std_err(exact_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Exact", color=exact_color, yscale=:log10)
    plot!(trΣ⁻¹_xdata, mean(trΣ⁻¹_ydata, dims=1)', ribbon = std_err(trΣ⁻¹_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "trΣ⁻¹", color=trΣ⁻¹_color, yscale=:log10)
    plot!(greedy_xdata, mean(greedy_ydata, dims=1)', ribbon = std_err(greedy_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Greedy", color=greedy_color, yscale=:log10)
    plot!(aspc_xdata, mean(aspc_ydata, dims=1)', ribbon = std_err(aspc_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "ASPC", color=aspc_color, yscale=:log10)
    plot!(mcts_xdata, mean(mcts_ydata, dims=1)', ribbon = std_err(mcts_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "MCTS", color=mcts_color, yscale=:log10)
    plot!(random_xdata, mean(random_ydata, dims=1)', ribbon = std_err(random_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Random", color=random_color, yscale=:log10)
    rt_plot = plot!(exact_xdata, data[1].timeout .* ones(size(exact_xdata)), label = "Timeout", color=:black, linestyle=:dash, linewidth=2, legend=false, yscale=:log10, dpi=500, widen=false, margin=5mm, size=(600,500))
    ###################################
    
    exact_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == Exact], (num_sims, length(exact_xdata)))
    trΣ⁻¹_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == trΣ⁻¹], (num_sims, length(trΣ⁻¹_xdata)))
    greedy_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == Greedy], (num_sims, length(greedy_xdata)))
    aspc_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == ASPC], (num_sims, length(aspc_xdata)))
    mcts_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == mcts], (num_sims, length(mcts_xdata)))
    random_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == random], (num_sims, length(random_xdata))) 
    
    plot(color_palette=:tab10)
    plot!(exact_xdata, mean(exact_ydata, dims=1)', ribbon = std_err(exact_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Exact", color=exact_color)
    plot!(trΣ⁻¹_xdata, mean(trΣ⁻¹_ydata, dims=1)', ribbon = std_err(trΣ⁻¹_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "trΣ", color=trΣ⁻¹_color, title="tr(Σ) vs. Graph Size", legend=false, framestyle=:box, widen=false, margin=5mm, size=(600,500))
    plot!(greedy_xdata, mean(greedy_ydata, dims=1)', ribbon = std_err(greedy_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Greedy", color=greedy_color)
    plot!(aspc_xdata, mean(aspc_ydata, dims=1)', ribbon = std_err(aspc_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "ASPC", color=aspc_color)
    plot!(mcts_xdata, mean(mcts_ydata, dims=1)', ribbon = std_err(mcts_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "MCTS", color=mcts_color)
    obj_plot = plot!(random_xdata, mean(random_ydata, dims=1)', ribbon = std_err(random_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Random", color=random_color, title="tr(Σ) vs. Graph Size", legend=false, framestyle=:box, widen=false, margin=5mm, size=(600,500))    
    
    plot([rt_plot, obj_plot]..., size=(1200, 500), layout=(1,2), margin=8mm, legend=false, widen=false)
    savefig("figures/paper/figure_2/a-ipp.pdf")
end

#####################################################################################
# Figure 3
#####################################################################################
function figure_3(load_data=false, data_path="data/")
    """
    Runtime and D-IPP objective as a function of the graph size.
    """
    grid_nodes = collect(123:-12:4).^2
    num_sims = 25
    edge_length = 1
    L = 0.01*edge_length # length scale 
    σ = 1.0
    objective = "D-IPP" 
    solution_time = 120.0
    rng = MersenneTwister(12345)
    m = 20
    B = 4*edge_length

    if load_data
        data = JLD2.load(data_path * "figure_3.jld2", "data")
    else
        data = []
        methods = [ASPC(), Exact(), trΣ⁻¹(), Greedy(), mcts(), random()]

        p = Progress(length(grid_nodes)*num_sims*length(methods))

        for (n_idx, n) in enumerate(grid_nodes)
            start = 1
            goal = n
            replan_rate = round(Int, 0.05 * B/edge_length * sqrt(n))#round(Int, 0.1 * B/edge_length * sqrt(n))

            # true map doesn't matter for non-adaptive objectives
            true_map = rand(rng, isqrt(n), isqrt(n))

            # Generate a grid graph
            Graph = build_graph(rng, data_path, n, m, edge_length, start, goal, objective)

            for (method_idx, method) in enumerate(methods)
                for i in 1:num_sims
                    shared_idx = (n_idx - 1) * length(methods) * num_sims + (method_idx - 1) * num_sims + i
                    # shared_idx = (n_idx-1)*length(methods)*num_sims + i
                    println("##########################################################################################")
                    println( string(shared_idx) * "/" * string(length(grid_nodes)*num_sims*length(methods)) * " grid_nodes " * string(n) * " run_type " * string(method))
                    println("##########################################################################################")

                    # Here we have to change only the Omega's 
                    omega_x = rand(rng, m)*edge_length
                    omega_y = rand(rng, m)*edge_length
                    Omega = hcat(omega_x, omega_y)
                    Graph = IPPGraph(Graph.G, Graph.start, Graph.goal, Graph.Theta, Omega, Graph.all_pairs_shortest_paths, Graph.distances, Graph.true_map, Graph.edge_length)

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

                    # Solve the IPP problem
                    val, t = @timed solve(ipp_problem, method)
                    path, objective_value = val
                    new_data = SimulationData(sim_number=shared_idx, run_type=method, n=n, m=m, B=B, L=L, replan_rate=replan_rate, timeout=solution_time, σ_min=1e-5, σ_max=σ, objVal=objective_value, y_hist=Vector{Float64}(), lower_bound=0.0, upper_bound=0.0, path=path, drills=Vector{Int}(), Omega=ipp_problem.Graph.Omega, runtime=t)
                    push!(data, new_data)

                    @show objective_value
                    @show t

                    # Plot the IPP problem
                    plot(ipp_problem, path, objective_value, t, "figures/paper/figure_3/runs/$(typeof(method))_$(n)n_$(objective)_$(i).pdf")
                    next!(p)
                    sleep(0.1)
                end
            end
        end
        JLD2.save(data_path * "figure_3.jld2", "data", data)
    end
    # Plotting 
    exact_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == Exact])
    trΣ⁻¹_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == trΣ⁻¹])
    greedy_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == Greedy])
    aspc_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == ASPC])
    mcts_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == mcts])
    random_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == random])

    exact_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == Exact], (num_sims, length(exact_xdata)))
    trΣ⁻¹_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == trΣ⁻¹], (num_sims, length(trΣ⁻¹_xdata)))
    greedy_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == Greedy], (num_sims, length(greedy_xdata)))
    aspc_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == ASPC], (num_sims, length(aspc_xdata)))
    mcts_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == mcts], (num_sims, length(mcts_xdata)))
    random_ydata = reshape([data[i].runtime for i in 1:length(data) if typeof(data[i].run_type) == random], (num_sims, length(random_xdata)))

    ###################################
    # Runtime vs. Number of Graph Nodes
    plot(color_palette=:tab10)
    plot!(exact_xdata, mean(exact_ydata, dims=1)', ribbon = std_err(exact_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Exact", color=exact_color, yscale=:log10)
    plot!(trΣ⁻¹_xdata, mean(trΣ⁻¹_ydata, dims=1)', ribbon = std_err(trΣ⁻¹_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "trΣ⁻¹", color=trΣ⁻¹_color, yscale=:log10)
    plot!(greedy_xdata, mean(greedy_ydata, dims=1)', ribbon = std_err(greedy_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Greedy", color=greedy_color, yscale=:log10)
    plot!(aspc_xdata, mean(aspc_ydata, dims=1)', ribbon = std_err(aspc_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "ASPC", color=aspc_color, yscale=:log10)
    plot!(mcts_xdata, mean(mcts_ydata, dims=1)', ribbon = std_err(mcts_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "MCTS", color=mcts_color, yscale=:log10)
    plot!(random_xdata, mean(random_ydata, dims=1)', ribbon = std_err(random_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Random", color=random_color, yscale=:log10)
    rt_plot = plot!(exact_xdata, data[1].timeout .* ones(size(exact_xdata)), label = "Timeout", color=:black, linestyle=:dash, linewidth=2, legend=false, yscale=:log10, dpi=500, widen=false, margin=5mm, size=(600,500))
    ###################################
    
    exact_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == Exact], (num_sims, length(exact_xdata)))
    trΣ⁻¹_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == trΣ⁻¹], (num_sims, length(trΣ⁻¹_xdata)))
    greedy_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == Greedy], (num_sims, length(greedy_xdata)))
    aspc_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == ASPC], (num_sims, length(aspc_xdata)))
    mcts_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == mcts], (num_sims, length(mcts_xdata)))
    random_ydata = reshape([data[i].objVal for i in 1:length(data) if typeof(data[i].run_type) == random], (num_sims, length(random_xdata))) 
    
    plot(color_palette=:tab10)
    plot!(exact_xdata, mean(exact_ydata, dims=1)', ribbon = std_err(exact_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Exact", color=exact_color)
    plot!(trΣ⁻¹_xdata, mean(trΣ⁻¹_ydata, dims=1)', ribbon = std_err(trΣ⁻¹_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "trΣ", color=trΣ⁻¹_color, title="logdet(Σ) vs. Graph Size", legend=false, framestyle=:box, widen=false, margin=5mm, size=(600,500))
    plot!(greedy_xdata, mean(greedy_ydata, dims=1)', ribbon = std_err(greedy_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Greedy", color=greedy_color)
    plot!(aspc_xdata, mean(aspc_ydata, dims=1)', ribbon = std_err(aspc_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "ASPC", color=aspc_color)
    plot!(mcts_xdata, mean(mcts_ydata, dims=1)', ribbon = std_err(mcts_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "MCTS", color=mcts_color)
    obj_plot = plot!(random_xdata, mean(random_ydata, dims=1)', ribbon = std_err(random_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Random", color=random_color, title="logdet(Σ) vs. Graph Size", legend=false, framestyle=:box, widen=false, margin=5mm, size=(600,500))    
    
    plot([rt_plot, obj_plot]..., size=(1200, 500), layout=(1,2), margin=8mm, legend=false, widen=false)
    savefig("figures/paper/figure_3/d-ipp.pdf")
end

#####################################################################################
# Figure 4
#####################################################################################
function figure_4()

end

#####################################################################################
# Figure 5
#####################################################################################
function figure_5()

end

#####################################################################################
# Figure 6
#####################################################################################
function figure_6()

end

#####################################################################################
# Figure 7
#####################################################################################
function figure_7()

end

#####################################################################################
# Figure 9
#####################################################################################
function figure_9()

end


# figure_1(true)
# figure_1()
figure_3()