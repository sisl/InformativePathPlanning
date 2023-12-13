function build_graph(rng, data_path::String, n::Int, m::Int, edge_length::Int, start::Int, goal::Int)
    G, Theta, Omega = build_G_Theta_Omega(rng, n, m, edge_length)
    dist = pairwise(Euclidean(), Theta', dims=2)

    # precompute all pairs shortest paths
    graph = []
    all_pairs_shortest_paths = []
    try # try loading them first
        println("Loading graph...")
        graph = JLD2.load(data_path * "/graph_cache/" * "$(n)_graph.jld2", "graph")
        all_pairs_shortest_paths = JLD2.load(data_path * "/graph_cache/" * "$(n)_all_pairs_shortest_paths.jld2", "all_pairs_shortest_paths")
        println("Loaded graph")
    catch
        println("Caught! Building graph and all_pairs_shortest_paths")
        graph = build_graph(G, n, Theta)
        all_pairs_shortest_paths = Graphs.floyd_warshall_shortest_paths(graph)
        JLD2.save(data_path * "/graph_cache/" * "$(n)_graph.jld2", "graph", graph)
        JLD2.save(data_path * "/graph_cache/" * "$(n)_all_pairs_shortest_paths.jld2", "all_pairs_shortest_paths", all_pairs_shortest_paths)
    end

    return IPPGraph(G, start, goal, Theta, Omega, all_pairs_shortest_paths, dist, edge_length)
end

function build_graph(G::Vector{Any}, n::Int, Theta::Matrix{Float64})
    idx = []
    for (v1, edges) in collect(enumerate(G))
        for v2 in edges
            push!(idx, (v1, v2)) 
        end
    end

    graph = SimpleWeightedGraph(n)

    for (v1, v2) in idx
        x1,y1 = Theta[v1, :]
        x2,y2 = Theta[v2, :]
        weight = norm([x1-x2, y1-y2])
        SimpleWeightedGraphs.add_edge!(graph, v1, v2, weight) 
    end

    return graph
end

function build_G_Theta_Omega(rng, n::Int, m::Int, edge_length::Int)
    # create observation vertex set (Theta)
    n_sqrt = isqrt(n)
    l = 0
    ux = n_sqrt - 1 # ensures edge lengths are 1
    uy = n_sqrt - 1 # ensures edge lengths are 1

    # grid is in [0, edge_length] x [0, edge_length]
    xg = collect((range(0, edge_length, n_sqrt))) #collect((LinRange(l, ux, n))) ./ m #collect(0:1//m:1)[2:end]#
    yg = collect((range(0, edge_length, n_sqrt))) #collect((LinRange(l, uy, n))) ./ m #collect(0:1//m:1)[2:end]#
    X, Y = meshgrid(xg, yg)
    Theta = hcat(X[:], Y[:])

    # create prediction location set (Omega)
    Omega = Theta[rand(rng, 1:n, m), :]

    # create adjacency list
    G = adjList(n)

    return G, Theta, Omega
end

# Julia doesn't have a built-in meshgrid function like numpy, 
# we will need to define our own
function meshgrid(x, y)
    nx, ny = length(x), length(y)
    x = reshape(x, nx, 1)
    y = reshape(y, 1, ny)
    return repeat(x, 1, ny), repeat(y, nx, 1)
end

function adjList(n::Int)
    n_sqrt = trunc(Int, sqrt(n))
    adj = []
    for v in 0:(n-1)
        edges = []
        y, x = divrem(v, n_sqrt)

        # correct for python 0 indexing 
        v += 1

        if (x+1) < n_sqrt
            push!(edges, v+1)
        end

        if (x-1) >= 0
            push!(edges, v-1)
        end

        if (y+1) < n_sqrt
            push!(edges, v+n_sqrt)
        end

        if (y-1) >= 0
            push!(edges, v-n_sqrt)
        end

        push!(adj, edges)
    end

    return adj
end
