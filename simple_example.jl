include("IPP.jl")

function run_simple_example()
    data_path = "/Users/joshuaott/InformativePathPlanning/data/"
    rng = MersenneTwister(12345)

    n = 25^2
    m = 20
    start = 1
    goal = n
    objective = "A-IPP"
    edge_length = 1
    B = 4*edge_length
    solution_time = 120.0
    replan_rate = round(Int, 0.1 * B/edge_length * sqrt(n))
    true_map = reshape(rand(rng, MvNormal(zeros(n), Diagonal(ones(n)*0.1))), (isqrt(n), isqrt(n)))

    # Generate a grid graph
    Graph = build_graph(rng, data_path, n, m, edge_length, start, goal)

    # Generate a measurement model
    σ = 0.1
    L = 0.1*edge_length # length scale 
    Σₓ = kernel(Graph.Theta, Graph.Theta, L)
    A = Matrix{Int}(I, n, m)
    measurement_model = MeasurementModel(σ, Σₓ, L, A)

    # Create an IPP problem
    ipp_problem = IPP(rng, n, m, Graph, measurement_model, objective, B, true_map, solution_time, replan_rate)

    # Solve the IPP problem
    val, t = @timed solve(ipp_problem)
    path, objective_value = val

    # Plot the IPP problem
    plot(ipp_problem, path, objective_value, t)
end

run_simple_example()