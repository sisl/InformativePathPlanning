using JLD2
using Random
using Statistics
using Distributions
using Plots
using KernelFunctions
using JLD2
using LinearAlgebra
using AbstractGPs

function build_map(rng::RNG, G::Vector{Vector{Int64}}, number_of_sample_types::Int, map_size::Tuple{Int, Int}) where {RNG<:AbstractRNG}
	sample_types = collect(0:(1/number_of_sample_types):(1-1/number_of_sample_types))
	init_map = rand(rng, sample_types, map_size[1], map_size[2])
	new_map = zeros(map_size)

	p_neighbors = 0.95

	for i in 1:(map_size[1]*map_size[2])
		if i == 1
			continue
		else
			if rand(rng) < p_neighbors
				neighbor_values = init_map[G[i]]
				new_map[i] = round(mean(neighbor_values),digits=1)
			else
				continue
			end
		end
	end

	return new_map
end

# function build_rand_maps()
#     i = 1
#     idx = 1
#     seed = 1234
#     while idx <= num_trials
#         @show i
#         rng = MersenneTwister(seed+i)

#         true_map = build_map(rng, number_of_sample_types, map_size_sboaippms)
#         JLD2.save(path_name * "/true_maps/true_map$(idx).jld", "true_map", true_map)
#         i += 1
#         idx += 1
#     end
# end

# function build_large_maps()
# 	i = 1
# 	idx = 1
# 	seed = 1234
# 	while idx <= num_trials
# 		@show i
# 		rng = MersenneTwister(seed+i)

# 		true_map = build_map(rng, number_of_sample_types, map_size_sboaippms)

# 		##########################################################################
# 		# GP
# 		##########################################################################
# 		L = length_scale*3
# 		gp = AbstractGPs.GP(with_lengthscale(SqExponentialKernel(), L))
# 		# build GP from true map and use posterior mean as the true map to get smoother true map

# 		for i in 1:length(true_map)
# 			# X = [X_query[i]]
# 			X = [[CartesianIndices(query_size)[i].I[1], CartesianIndices(query_size)[i].I[2]]] ./ 10 #./100

# 			y = [true_map[i]]
# 			gp = AbstractGPs.posterior(gp(X, 0.1), y)
# 		end

# 		X_plot_query = [[i,j] for i = range(0, 1, length=(640)), j = range(0, 1, length=(640))]

# 		X_plot_query = reshape(X_plot_query, size(X_plot_query)[1]*size(X_plot_query)[2])

# 		plot_map = reshape(mean(gp(X_plot_query)), (round(Int, sqrt(length(X_plot_query))),round(Int, sqrt(length(X_plot_query)))))
# 		true_map = plot_map

# 		JLD2.save(path_name * "/true_maps/true_map$(idx).jld", "true_map", true_map)
# 		heatmap(true_map)
# 		savefig(path_name * "/true_maps/true_map$(idx).png")

# 		i += 1
# 		idx += 1
# 	end
# end

function build_gp_maps(rng::Random.AbstractRNG, G::Vector{Vector{Int64}}, N::Int, num_sims::Int, Theta::Matrix{Float64}, L::Float64, map_path::String)
    idx = 1
    number_of_sample_types = 10
    map_size = (isqrt(N), isqrt(N))
    while idx <= num_sims
        @show idx

        true_map = build_map(rng, G, number_of_sample_types, map_size)

		##########################################################################
		# GP
		##########################################################################
        # build GP from true map and use posterior mean as the true map to get smoother true map
		gp = AbstractGPs.GP(with_lengthscale(SqExponentialKernel(), L))

        # take 10% of the points as sampled locations to build gp map
        X = [Theta[i, :] for i in rand(rng, 1:N, round(Int, N/10))]
        Y = [true_map[i] for i in rand(rng, 1:N, round(Int, N/10))]
        X_query = [Theta[i, :] for i in 1:N]
        gp = AbstractGPs.posterior(gp(X, 0.1), Y)
    
        plot_map = reshape(mean(gp(X_query)), (isqrt(N), isqrt(N)))
        true_map = plot_map

        JLD2.save(map_path * "true_map_$(N)_$(idx).jld", "true_map", true_map)
        plot_scale = collect(range(0,100,isqrt(N)))
        heatmap(plot_scale, plot_scale, true_map, clim=(0,1))
		savefig(map_path * "true_map_$(N)_$(idx).png")

        idx += 1
    end
end

function build_gp_map(rng::Random.AbstractRNG, G::Vector{Vector{Int64}}, N::Int, Theta::Matrix{Float64}, L::Float64, map_path::String, idx::Int)
    number_of_sample_types = 10
    map_size = (isqrt(N), isqrt(N))

    true_map = build_map(rng, G, number_of_sample_types, map_size)

    ##########################################################################
    # GP
    ##########################################################################
    # build GP from true map and use posterior mean as the true map to get smoother true map
    gp = AbstractGPs.GP(with_lengthscale(SqExponentialKernel(), L))

    N_plot = N #round(Int, (isqrt(N)/2)^2)
    X = [Theta[i, :] for i in 1:N_plot]
    Y = [true_map[i] for i in 1:N_plot]
    gp = AbstractGPs.posterior(gp(X, 1.0), Y)

    plot_map = reshape(mean(gp(X)), (isqrt(N_plot), isqrt(N_plot)))
    true_map = plot_map

    JLD2.save(map_path * "true_map_$(N)_$(idx).jld", "true_map", true_map)
    plot_scale = collect(range(0,100,isqrt(N)))
    heatmap(plot_scale, plot_scale, true_map, clim=(0,1))
    savefig(map_path * "true_map_$(N)_$(idx).png")

    return true_map
end

function build_maps()
    for n in collect(123:-12:4).^2#collect(10:-1:2).^2#collect(100:-10:4).^2
        edge_length = 1#100
        m = 20
        # NOTE: we can use different length scales here for map generation
        L = 0.08*edge_length
        rng = MersenneTwister(1234567)
        num_sims = 25
        G, Theta, Omega = build_G_Theta_Omega(rng, n, m, edge_length)
        # G, Theta, Omega = build_graph(rng, N, 20)
        build_gp_maps(rng, G, n, num_sims, Theta, L, "../../data/maps/")
    end
end