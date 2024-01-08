include("IPP.jl")
include("utilities/refinement.jl")

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
    EI_hist::Vector{Float64}
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
                    new_data = SimulationData(sim_number=shared_idx, run_type=method, n=n, m=m, B=B, L=L, replan_rate=replan_rate, timeout=solution_time, σ_min=1e-5, σ_max=σ, objVal=objective_value, y_hist=Vector{Float64}(), EI_hist=Vector{Float64}(), lower_bound=0.0, upper_bound=0.0, path=path, drills=Vector{Int}(), Omega=ipp_problem.Graph.Omega, runtime=t)
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
        methods = [ASPC(), Exact(), trΣ⁻¹(), Greedy(), mcts(), random()]

        p = Progress(length(grid_nodes)*num_sims*length(methods))

        for (n_idx, n) in enumerate(grid_nodes)
            start = 1
            goal = n
            replan_rate = round(Int, 0.05 * B/edge_length * sqrt(n))#round(Int, 0.1 * B/edge_length * sqrt(n))

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
                    new_data = SimulationData(sim_number=shared_idx, run_type=method, n=n, m=m, B=B, L=L, replan_rate=replan_rate, timeout=solution_time, σ_min=1e-5, σ_max=σ, objVal=objective_value, y_hist=Vector{Float64}(), EI_hist=Vector{Float64}(), lower_bound=0.0, upper_bound=0.0, path=path, drills=Vector{Int}(), Omega=ipp_problem.Graph.Omega, runtime=t)
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
                    new_data = SimulationData(sim_number=shared_idx, run_type=method, n=n, m=m, B=B, L=L, replan_rate=replan_rate, timeout=solution_time, σ_min=1e-5, σ_max=σ, objVal=objective_value, y_hist=Vector{Float64}(), EI_hist=Vector{Float64}(), lower_bound=0.0, upper_bound=0.0, path=path, drills=Vector{Int}(), Omega=ipp_problem.Graph.Omega, runtime=t)
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
struct Fig5Data
    sim_data::SimulationData
    mcts_drills::Vector{Int64}
    mcts_objective::Float64
    cvx_drills::Vector{Int64}
    cvx_objective::Float64
end

function figure_5(load_data=false, data_path="data/")
    """
    Multimodal sensing comparison between MCTS and proposed approach.
    """
    grid_nodes = collect(123:-12:4).^2
    num_sims = 25
    edge_length = 1
    L = 0.01*edge_length # length scale 
    σ = 1.0
    solution_time = 120.0
    rng = MersenneTwister(12345)
    m = 20
    B = 4*edge_length

    plts = []

    for obj in ["A-IPP", "D-IPP"]
        if load_data
            data = JLD2.load(data_path * "figure_5_$(obj).jld2", "data")
        else
            data = []
            methods = [mcts()]

            p = Progress(length(grid_nodes)*num_sims*length(methods))

            for (n_idx, n) in enumerate(grid_nodes)
                start = 1
                goal = n
                replan_rate = round(Int, 0.05 * B/edge_length * sqrt(n))#round(Int, 0.1 * B/edge_length * sqrt(n))

                # Generate a grid graph
                Graph = build_graph(rng, data_path, n, m, edge_length, start, goal, obj)

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

                        # Multimodal Sensing
                        σ_max = σ
                        σ_min = 1e-5
                        k = 3

                        # Create an IPP problem
                        ipp_problem = IPP(rng, n, m, Graph, measurement_model, obj, B, solution_time, replan_rate)
                        mmipp = MultimodalIPP(ipp_problem, σ_min, σ_max, k)

                        # Solve the IPP problem
                        val, t = @timed solve(mmipp, mcts())
                        path, objective_value, drills = val
                        y_hist = zeros(length(path)) # y_hist is not used for A-IPP or D-IPP

                        objective_value = objective(mmipp, path, y_hist, drills)

                        sim_data = SimulationData(sim_number=shared_idx, run_type=method, n=n, m=m, B=B, L=L, replan_rate=replan_rate, timeout=solution_time, σ_min=σ_min, σ_max=σ_max, objVal=objective_value, y_hist=Vector{Float64}(), EI_hist=Vector{Float64}(), lower_bound=0.0, upper_bound=0.0, path=path, drills=Vector{Int}(), Omega=ipp_problem.Graph.Omega, runtime=t)
                        mcts_drills = deepcopy(drills)
                        mcts_objective = deepcopy(objective_value)

                        # Now run cvx drill selection
                        if obj == "A-IPP"
                            cvx_drills = run_a_optimal_sensor_selection(mmipp, unique(path))
                        elseif obj == "D-IPP"
                            cvx_drills = run_d_optimal_sensor_selection(mmipp, unique(path))
                        else
                            @error("Objective not supported")
                        end
                        cvx_objective = objective(mmipp, path, y_hist, cvx_drills)
                        new_data = Fig5Data(sim_data, mcts_drills, mcts_objective, cvx_drills, cvx_objective)
                        push!(data, new_data)

                        @show mcts_objective
                        @show cvx_objective
                        @show t

                        # Plot the IPP problem
                        plot(mmipp, path, mcts_drills, cvx_drills, objective_value, t, "figures/paper/figure_5/$(obj)_runs/$(typeof(method))_$(n)n_$(obj)_$(i).pdf")
                        next!(p)
                        sleep(0.1)
                    end
                end
            end
            JLD2.save(data_path * "figure_5_$(obj).jld2", "data", data)
        end

        # Plotting
        xdata = unique([data[i].sim_data.n for i in 1:length(data)])
        mcts_mean_obj_hist = []
        mcts_std_err_obj_hist = []
        cvx_mean_obj_hist = []
        cvx_std_err_obj_hist = []

        for n in xdata
            mcts_objectives = [data[i].mcts_objective for i in 1:length(data) if data[i].sim_data.n == n]
            cvx_objectives = [data[i].cvx_objective for i in 1:length(data) if data[i].sim_data.n == n]
            push!(mcts_mean_obj_hist, mean(mcts_objectives))
            push!(mcts_std_err_obj_hist, std_err(mcts_objectives, num_sims)[1])
            push!(cvx_mean_obj_hist, mean(cvx_objectives))
            push!(cvx_std_err_obj_hist, std_err(cvx_objectives, num_sims)[1])
        end

        title = obj == "A-IPP" ? "tr(Σ) vs. Graph Size" : "logdet(Σ) vs. Graph Size"
        plot(xdata, mcts_mean_obj_hist, ribbon = mcts_std_err_obj_hist, fillalpha = 0.2, label="MCTS", title=title, color=mcts_color, color_palette=:tab10, framestyle=:box)
        plt_obj = plot!(xdata, cvx_mean_obj_hist, ribbon = cvx_std_err_obj_hist, fillalpha = 0.2, label="Convex", title=title, color=aspc_color, color_palette=:tab10, framestyle=:box, legend=false, widen=false, size=(600,500), margin=5mm, xticks=[0.0, 5e3, 10e3, 15e3])
        push!(plts, plt_obj)
    end
    plot(plts..., layout=(1,2), size=(1200, 500), margin=8mm, legend=false, widen=false)
    savefig("figures/paper/figure_5/cvx_drill_placement.pdf")
end



#####################################################################################
# Figure 6
#####################################################################################
function figure_6(load_data=false, data_path="data/")
    """
    Computes optimality gap for both the A and D-IPP objectives
    """
    n = 40^2
    num_sims = 25
    edge_length = 1
    L = 0.01*edge_length # length scale 
    σ = 1.0
    solution_time = 120.0
    rng = MersenneTwister(12345)
    m = 20
    budgets=collect(4:4:round(Int, n / sqrt(n)))*edge_length
    a_opt_plt = Nothing
    d_opt_plt = Nothing

    for obj in ["A-IPP", "D-IPP"]
        if load_data
            data = JLD2.load(data_path * "figure_6_$(obj).jld2", "data")
        else
            data = []
            methods = [ASPC()]
    
            p = Progress(length(budgets)*num_sims*length(methods))

            for (b_idx, B) in enumerate(budgets)
                start = 1
                goal = n
                replan_rate = round(Int, 0.05 * B/edge_length * sqrt(n))#round(Int, 0.1 * B/edge_length * sqrt(n))
    
                # Generate a grid graph
                Graph = build_graph(rng, data_path, n, m, edge_length, start, goal, obj)

                for (method_idx, method) in enumerate(methods)
                    for i in 1:num_sims
                        shared_idx = (b_idx - 1) * length(methods) * num_sims + (method_idx - 1) * num_sims + i
                        println("##########################################################################################")
                        println( string(shared_idx) * "/" * string(length(budgets)*num_sims*length(methods)) * " budgets " * string(B) * " run_type " * string(method))
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
                        ipp_problem = IPP(rng, n, m, Graph, measurement_model, obj, B, solution_time, replan_rate)
    
                        # Solve the IPP problem
                        val, t = @timed solve(ipp_problem, method)
                        path, objective_value = val
                        upper_bound = objective_value
                        lower_bound = relax(ipp_problem)
                        new_data = SimulationData(sim_number=shared_idx, run_type=method, n=n, m=m, B=B, L=L, replan_rate=replan_rate, timeout=solution_time, σ_min=1e-5, σ_max=σ, objVal=objective_value, y_hist=Vector{Float64}(), EI_hist=Vector{Float64}(), lower_bound=lower_bound, upper_bound=upper_bound, path=path, drills=Vector{Int}(), Omega=ipp_problem.Graph.Omega, runtime=t)
                        push!(data, new_data)
    
                        @show objective_value
                        @show upper_bound
                        @show lower_bound
                        @show t
    
                        # Plot the IPP problem
                        plot(ipp_problem, path, objective_value, t, "figures/paper/figure_6/$(obj)_runs/$(typeof(method))_$(B)B_$(i).pdf")
                        next!(p)
                        sleep(0.1)
                    end
                end
            end
            JLD2.save(data_path * "figure_6_$(obj).jld2", "data", data)
        end

        # Plotting
        xdata = unique([data[i].B for i in 1:length(data)])

        upper_bound = reshape([data[i].upper_bound for i in 1:length(data)], (num_sims, length(xdata)))
        lower_bound = reshape([data[i].lower_bound for i in 1:length(data)], (num_sims, length(xdata)))

        plot(color_palette=:tab10)
        title = obj == "A-IPP" ? "tr(Σ)" : "logdet(Σ)"

        if obj == "A-IPP"
            δ = (1/m)*(upper_bound - lower_bound) ./ ((1/m)*lower_bound)
            plot(xdata ./ edge_length, (mean(δ, dims=1)'), ribbon = (std_err(δ, num_sims)), fillalpha = 0.2, xlabel = "Budget", title=title, label = "Gap", color=aspc_color, legend=false, framestyle=:box, widen=false, size=(600,500), margin=5mm)
            # plot!(xdata ./ edge_length, (mean(upper_bound, dims=1)' - mean(lower_bound, dims=1)') ./ abs.(mean(lower_bound, dims=1)'), ribbon = (std_err(upper_bound, num_sims) - std_err(lower_bound, num_sims)) ./ abs.(mean(lower_bound, dims=1)'), fillalpha = 0.2, xlabel = "Budget", ylabel = "Gap", title=title, label = "Gap", color=:red, legend=false, framestyle=:box, widen=false, size=(600,500), margin=5mm)
        else
            δ = (1/m)*(upper_bound - lower_bound)
            plot(xdata ./ edge_length,  (mean(exp.(δ), dims=1)') , ribbon = std_err(exp.(δ), num_sims), fillalpha = 0.2, xlabel = "Budget", title=title, label = "Gap", color=aspc_color, legend=false, framestyle=:box, widen=false, size=(600,500), margin=5mm, ylims=(1.0, maximum(mean(exp.(δ), dims=1)')))
            # plot(xdata ./ edge_length, exp.( (1/m) * (mean(upper_bound, dims=1)' - mean(lower_bound, dims=1)' )), ribbon = exp.( (1/m) * (std_err(upper_bound, num_sims) - std_err(lower_bound, num_sims))), fillalpha = 0.2, xlabel = "Budget", ylabel = "Gap", title=title, label = "Gap", color=aspc_color, legend=false, framestyle=:box, widen=false, size=(600,500), margin=5mm)
        end

        if obj == "A-IPP"
            a_opt_plt = plot!()
        else
            d_opt_plt = plot!()
        end
    end
    gap_plt = [a_opt_plt, d_opt_plt]
    plot(gap_plt..., size=(1200, 500), layout=(1,2), margin=7mm, legend=false)#, #title="", xlabel="", ylabel="")
    savefig("figures/paper/figure_6/gap.pdf")
end

#####################################################################################
# Figure 7
#####################################################################################
function figure_7(load_data=false, data_path="data/")
    """
    Performs greedy swapping procedure as outlined by Joshi and Boyd
    """

    num_sims = 25
    rng = MersenneTwister(12345)
    a_opt_plts = Nothing
    d_opt_plts = Nothing

    for obj in ["A-IPP", "D-IPP"]
        data = obj == "A-IPP" ? JLD2.load(data_path * "figure_2.jld2", "data") : JLD2.load(data_path * "figure_3.jld2", "data")

        if load_data
            refined_data = JLD2.load(data_path * "figure_7_refined.jld2", "data")
        else
            refined_paths, refined_obj_vals = run_refinement(rng, data, obj)

            refined_data = []
            for i in 1:eachindex(data)
                new_data = SimulationData(sim_number=data[i].sim_number, run_type=data[i].run_type, n=data[i].n, m=data[i].m, B=data[i].B, L=data[i].L, replan_rate=data[i].replan_rate, timeout=data[i].timeout, σ_min=data[i].σ_min, σ_max=data[i].σ_max, objVal=refined_obj_vals[i], y_hist=Vector{Float64}(), EI_hist=Vector{Float64}(), lower_bound=0.0, upper_bound=0.0, path=refined_paths[i], drills=Vector{Int}(), Omega=data[i].Omega, runtime=data[i].runtime)
                push!(refined_data, new_data)
            end

            JLD2.save(data_path * "figure_7.jld2", "data", data)
        end

        # Plotting
        exact_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == Exact])
        trΣ⁻¹_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == trΣ⁻¹])
        greedy_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == Greedy])
        aspc_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == ASPC])
        mcts_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == mcts])
        random_xdata = unique([data[i].n for i in 1:length(data) if typeof(data[i].run_type) == random])
        
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
        unrefined_plt = plot!(random_xdata, mean(random_ydata, dims=1)', ribbon = std_err(random_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Random", color=random_color, title="tr(Σ) vs. Graph Size", legend=false, framestyle=:box, widen=false, margin=5mm, size=(600,500))
        
        exact_ydata = reshape([refined_data[i].objVal for i in 1:length(refined_data) if typeof(refined_data[i].run_type) == Exact], (num_sims, length(exact_xdata)))
        trΣ⁻¹_ydata = reshape([refined_data[i].objVal for i in 1:length(refined_data) if typeof(refined_data[i].run_type) == trΣ⁻¹], (num_sims, length(trΣ⁻¹_xdata)))
        greedy_ydata = reshape([refined_data[i].objVal for i in 1:length(refined_data) if typeof(refined_data[i].run_type) == Greedy], (num_sims, length(greedy_xdata)))
        aspc_ydata = reshape([refined_data[i].objVal for i in 1:length(refined_data) if typeof(refined_data[i].run_type) == ASPC], (num_sims, length(aspc_xdata)))
        mcts_ydata = reshape([refined_data[i].objVal for i in 1:length(refined_data) if typeof(refined_data[i].run_type) == mcts], (num_sims, length(mcts_xdata)))
        random_ydata = reshape([refined_data[i].objVal for i in 1:length(refined_data) if typeof(refined_data[i].run_type) == random], (num_sims, length(random_xdata)))

        plot(color_palette=:tab10)
        plot!(exact_xdata, mean(exact_ydata, dims=1)', ribbon = std_err(exact_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Exact", color=exact_color)
        plot!(trΣ⁻¹_xdata, mean(trΣ⁻¹_ydata, dims=1)', ribbon = std_err(trΣ⁻¹_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "trΣ", color=trΣ⁻¹_color, title="tr(Σ) vs. Graph Size", legend=false, framestyle=:box, widen=false, margin=5mm, size=(600,500))
        plot!(greedy_xdata, mean(greedy_ydata, dims=1)', ribbon = std_err(greedy_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Greedy", color=greedy_color)
        plot!(aspc_xdata, mean(aspc_ydata, dims=1)', ribbon = std_err(aspc_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "ASPC", color=aspc_color)
        plot!(mcts_xdata, mean(mcts_ydata, dims=1)', ribbon = std_err(mcts_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "MCTS", color=mcts_color)
        refined_plt = plot!(random_xdata, mean(random_ydata, dims=1)', ribbon = std_err(random_ydata, num_sims), fillalpha = 0.2, xlabel = "Number of Graph Nodes", label = "Random", color=random_color, title="tr(Σ) vs. Graph Size", legend=false, framestyle=:box, widen=false, margin=5mm, size=(600,500))

        if obj == "A-IPP"
            a_opt_plts = [unrefined_plt, refined_plt]
        else
            d_opt_plts = [unrefined_plt, refined_plt]
        end
    end
    plot([d_opt_plts..., a_opt_plts...]..., size=(800, 800), layout=(2,2), margin=5mm, legend=false, title="", xlabel="", ylabel="", xticks =[0.0, 5e3, 10e3, 15e3])

end

#####################################################################################
# Figure 9
#####################################################################################
function figure_9(load_data=false, data_path="data/")
    """
    Runtime and Expected Improvement adaptive objective as a function of the graph size.
    """
    grid_nodes = [123, 111, 75].^2
    num_sims = 25
    edge_length = 1
    L = 0.01*edge_length # length scale 
    # NOTE: for expected improvement σ_max=1.0 is quite noisy. σ_max=0.1 is more reasonable for EI use cases
    # for A and D-optimal σ_max=1.0 is fine since we are only focused on sensor distribution, not on the actual values received
    σ = 0.1
    objective = "expected_improvement" 
    solution_time = 120.0
    rng = MersenneTwister(12345)
    m = 20
    B = 4*edge_length

    if load_data
        data = JLD2.load(data_path * "figure_9.jld2", "data")
    else
        data = []
        methods = [Greedy(), ASPC(), random(), mcts()]

        p = Progress(length(grid_nodes)*num_sims*length(methods))

        for (n_idx, n) in enumerate(grid_nodes)
            start = 1
            goal = n
            replan_rate = round(Int, 0.02 * B/edge_length * sqrt(n))#round(Int, 0.1 * B/edge_length * sqrt(n))

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
                    true_map = load(data_path * "maps/true_map_$(n)_$(i).jld")["true_map"]
                    Graph = IPPGraph(Graph.G, Graph.start, Graph.goal, Graph.Theta, Omega, Graph.all_pairs_shortest_paths, Graph.distances, true_map, Graph.edge_length)

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
                    path, objective_value, y_hist = val
                    EI_hist = compute_adaptive_obj_hist(ipp_problem, y_hist, path)

                    new_data = SimulationData(sim_number=shared_idx, run_type=method, n=n, m=m, B=B, L=L, replan_rate=replan_rate, timeout=solution_time, σ_min=1e-5, σ_max=σ, objVal=objective_value, y_hist=y_hist, EI_hist=EI_hist, lower_bound=0.0, upper_bound=0.0, path=path, drills=Vector{Int}(), Omega=ipp_problem.Graph.Omega, runtime=t)
                    push!(data, new_data)

                    @show objective_value
                    @show t

                    # Plot the IPP problem
                    plot(ipp_problem, path, objective_value, t, "figures/paper/figure_9/runs/$(typeof(method))_$(n)n_$(objective)_$(i).pdf")
                    next!(p)
                    sleep(0.1)
                end
            end
        end
        JLD2.save(data_path * "figure_9.jld2", "data", data)
    end
    
    # Plotting 
    plots = []
    Ns = [123, 111, 75].^2#unique([data[i].n for i in 1:length(data)])
    for N in Ns
        plot()

        min_plot_length = minimum([length(data[i].EI_hist) for i in 1:length(data) if data[i].n == N])
        for run_type in [random, Greedy, mcts, ASPC]
            EI_hist = [data[i].EI_hist for i in 1:length(data) if typeof(data[i].run_type) == run_type && data[i].n == N]

            max_length = maximum([length(EI_hist[i]) for i in 1:length(EI_hist)])
            for i in 1:num_sims
                while length(EI_hist[i]) != max_length
                    push!(EI_hist[i], EI_hist[i][end])
                end
            end
            mean_EI = reshape(mean(hcat(EI_hist...), dims=2), (max_length,))
            std_err_EI = reshape(std_err(hcat(EI_hist...)', num_sims), (max_length,))

            color = run_type == ASPC ? aspc_color : run_type == Exact ? exact_color : run_type == mcts ? mcts_color : run_type == random ? random_color : run_type == Greedy ? greedy_color : run_type == DuttaMIP ? mip_color : run_type == trΣ⁻¹ ? trΣ⁻¹_color : relaxed_color
            label = run_type == ASPC ? "ASPC" : run_type == Exact ? "Exact" : run_type == mcts ? "MCTS" : run_type == random ? "Random" : run_type == Greedy ? "Greedy" : run_type == DuttaMIP ? "Dutta MIP" : run_type == trΣ⁻¹ ? "TrΣ⁻¹" : relaxed_color
            plot!(mean_EI[1:min_plot_length] ./ m, ribbon = std_err_EI[1:min_plot_length] ./m, fillalpha = 0.2, label=label, title="N = $N", color=color, color_palette=:tab10, framestyle=:box, widen=false)
            # plot!(mean_EI[1:min_plot_length] ./ N, ribbon = std_err_EI[1:min_plot_length] ./N, fillalpha = 0.2, label=label, title="N = $N", color=color, color_palette=:tab10, framestyle=:box, widen=false)
        end
        plt = plot!()
        push!(plots, plt)
    end
    plot(plots..., layout=(1, 3), size=(1800, 500), legend=false, margin=7mm)
   
    savefig("figures/paper/figure_9/expected_improvement.pdf")
end

# figure_1(true)
# figure_1()
# figure_2()
# figure_3()
# figure_5(true)
# figure_6(true)
figure_7()
# figure_9(true)