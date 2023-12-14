include("IPP.jl")

function run_simple_example()
    data_path = "/Users/joshuaott/InformativePathPlanning/data/"
    rng = MersenneTwister(12345)

    n = 10^2
    m = 20
    start = 1
    goal = n
    objective = "D-IPP"#"expected_improvement"
    edge_length = 1
    B = 4*edge_length
    solution_time = 120.0
    replan_rate = round(Int, 0.1 * B/edge_length * sqrt(n))
    true_map = reshape(rand(rng, MvNormal(zeros(n), Diagonal(ones(n)*0.1))), (isqrt(n), isqrt(n)))

    # Generate a grid graph
    Graph = build_graph(rng, data_path, n, m, edge_length, start, goal, objective)

    # Generate a measurement model
    σ = 1.0
    L = 0.01*edge_length # length scale 
    Σₓ = kernel(Graph.Omega, Graph.Omega, L) # = K(X⁺, X⁺)
    ϵ = Matrix{Float64}(I, size(Σₓ))*1e-6 # Add a Small Constant to the Diagonal (Jitter): This is a common technique to improve the numerical stability of a kernel matrix. 
    Σₓ⁻¹ = inv(Σₓ + ϵ)
    KX⁺X = kernel(Graph.Omega, Graph.Theta, L) # = K(X⁺, X)
    Aᵀ = Σₓ⁻¹ * KX⁺X
    A = Aᵀ'
    measurement_model = MeasurementModel(σ, Σₓ, Σₓ⁻¹, L, A)

    # Create an IPP problem
    ipp_problem = IPP(rng, n, m, Graph, measurement_model, objective, B, solution_time, replan_rate)

    # Solve the IPP problem
    val, t = @timed solve(ipp_problem)
    path, objective_value = val

    # Plot the IPP problem
    plot(ipp_problem, path, objective_value, t)

end

run_simple_example()