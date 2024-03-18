function create_nodes_of_interest(points_of_interest::Matrix{Float64},
                                  Theta::Matrix{Float64})
    noi = Vector{Int64}()
    for i in 1:size(points_of_interest, 1)
        p = points_of_interest[i, :]
        closest_node = argmin([norm(Theta[j, :] - p) for j in 1:size(Theta, 1)])
        push!(noi, closest_node)
    end
    return noi
end

function run_poi_example()
    data_path = "../data/"
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
    ipp_problem = IPP(rng, n, m, Graph, measurement_model, objective, B, solution_time, replan_rate, "open")

    points_of_interest = [0.5 0.5; 0.25 0.75; 0.75 0.25]
    noi = create_nodes_of_interest(points_of_interest, Graph.Theta)
    poiipp = PointsOfInterestIPP(ipp_problem, noi)

    # Solve the IPP problem
    val, t = @timed solve(poiipp, Exact())
    path, objective_value = val
end

function run_poi_multimodal_example()
    data_path = "../data/"
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
    k = 3

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
    ipp_problem = IPP(rng, n, m, Graph, measurement_model, objective, B, solution_time, replan_rate, "open")
    
    # Points of interest
    points_of_interest = [0.5 0.5; 0.25 0.75; 0.75 0.25]
    noi = create_nodes_of_interest(points_of_interest, Graph.Theta)
    poiipp = PointsOfInterestIPP(ipp_problem, noi)

    mmipp = MultimodalPOIIPP(poiipp, σ_min, σ_max, k)

    # Solve the IPP problem
    val, t = @timed solve(mmipp, Exact())
    path, drills, objective_value = val

    # Plot the IPP problem
    plot(mmipp, path, drills, objective_value, t)

end