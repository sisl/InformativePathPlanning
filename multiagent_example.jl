include("IPP.jl")

function run_multiagent_example()
    data_path = "/Users/joshuaott/InformativePathPlanning/data/"
    rng = MersenneTwister(123456)

    plot_gif = true
    obstacles = true
    M = 3 # number of agents 
    n = 20^2
    m = 20
    start = 1
    goal = n
    objective = "A-IPP"#"expected_improvement"
    edge_length = 1
    B = round(Int, 3*edge_length)
    solution_time = 120.0
    replan_rate = 1#round(Int, 0.05 * B/edge_length * sqrt(n))
    true_map = rand(rng, isqrt(n), isqrt(n))

    # Generate a grid graph
    if obstacles
        Graph, centers, radii = build_graph(rng, data_path, n, m, edge_length, start, goal, objective, 1, obstacles)
    else
        Graph = build_graph(rng, data_path, n, m, edge_length, start, goal, objective, 1, obstacles)
    end

    # Generate a measurement model
    σ = 1.0
    L = 0.07*edge_length # length scale 
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
    mipp = MultiagentIPP(ipp_problem, M)

    # Solve the IPP problem
    paths, t = @timed solve(mipp, ASPC(), plot_gif, centers, radii)
end

run_multiagent_example()