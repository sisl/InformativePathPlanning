function run_simple_example(solver_type::String="open")
    # data_path is one directory up from the current directory lets use .. to go up one directory
    data_path = "../data/"
    rng = Random.MersenneTwister(12345)

    n = 10^2
    m = 20
    start = 1
    goal = n
    objective = "A-IPP"#"expected_improvement"
    edge_length = 1
    B = 4*edge_length
    solution_time = 120.0
    replan_rate = round(Int, 0.05 * B/edge_length * sqrt(n))
    true_map = Random.rand(rng, isqrt(n), isqrt(n))

    # Generate a grid graph
    Graph = build_graph(rng, data_path, n, m, edge_length, start, goal, objective)

    # Generate a measurement model
    σ = 1.0
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
    ipp_problem = IPP(rng, n, m, Graph, measurement_model, objective, B, solution_time, replan_rate, solver_type)

    # Solve the IPP problem
    val, t = @timed solve(ipp_problem, ASPO())
    path, objective_value = val

    @show relax(ipp_problem)

    # Plot the IPP problem
    plot(ipp_problem, path, objective_value, t)

end