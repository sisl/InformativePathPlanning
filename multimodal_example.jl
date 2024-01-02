include("IPP.jl")

function run_multimodal_example()
    data_path = "/Users/joshuaott/InformativePathPlanning/data/"
    rng = MersenneTwister(12345)

    n = 10^2
    m = 20
    start = 1
    goal = n
    objective = "A-IPP"#"expected_improvement"
    edge_length = 1
    B = 4*edge_length
    solution_time = 120.0
    replan_rate = round(Int, 0.05 * B/edge_length * sqrt(n))
    true_map = rand(rng, isqrt(n), isqrt(n))

    # Generate a grid graph
    Graph = build_graph(rng, data_path, n, m, edge_length, start, goal, objective)

    # Multimodal Sensing
    σ = 1.0
    σ_max = σ
    σ_min = 1e-5
    k = 2

    # Generate a measurement model
    L = 0.01*edge_length # length scale 
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
    mmipp = MultimodalIPP(ipp_problem, σ_min, σ_max, k)

    # Solve the IPP problem
    val, t = @timed solve(mmipp, ASPC())
    path, objective_value = val

    # Plot the IPP problem
    plot(ipp_problem, path, objective_value, t)

end

run_multimodal_example()