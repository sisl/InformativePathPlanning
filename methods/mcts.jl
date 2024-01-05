include("SBO_AIPPMS/CustomGP.jl")
include("SBO_AIPPMS/rover_pomdp.jl")
include("SBO_AIPPMS/belief_mdp.jl")
using Random
using BasicPOMCP
using POMDPs
using Statistics
using Distributions
using Plots
using KernelFunctions
using MCTS
using DelimitedFiles
using JLD2
using ProgressMeter

POMDPs.isterminal(bmdp::BeliefMDP, b::RoverBelief) = isterminal(bmdp.pomdp, b)

################################################################################
# Running Trials
################################################################################

function run_rover_bmdp(rng::RNG, run_mcts, run_random, bmdp::BeliefMDP, policy, isterminal::Function, all_pairs_shortest_paths, timeout::Float64, rollout_depth, rollout_iterations) where {RNG<:AbstractRNG}

    belief_state = initialstate(bmdp).val
    # belief_state.cost_expended = 1.0 # set initial cost to 1

    # belief_state = initial_belief_state(bmdp, rng)
	state_hist = [deepcopy(belief_state.pos)]
	gp_hist = [deepcopy(belief_state.location_belief)]
	action_hist = []
	reward_hist = []
	total_reward_hist = []
	total_planning_time = 0
	total_iterations = 0
	iter = rollout_iterations

    total_reward = 0.0
    while true
		@show belief_state.pos
		@show belief_state.cost_expended
		@show shortest_path_to_goal(bmdp.pomdp, belief_state.pos)
		@show actions(bmdp.pomdp, belief_state)

        a, t = @timed policy(belief_state)

		total_planning_time += t

		if run_mcts && belief_state.cost_expended != 0.0
			# scale number of iterations based on planning time so far
			total_iterations += iter
			cost_left = bmdp.pomdp.cost_budget - belief_state.cost_expended
			cost_per_step = belief_state.cost_expended / length(state_hist)
			steps_left = cost_left / cost_per_step
			time_left = timeout - total_planning_time
			time_per_iter = total_planning_time/total_iterations
			iterations_left = time_left/time_per_iter
			iter = round(Int, iterations_left / steps_left)

			@show total_iterations
			@show cost_left
			@show cost_per_step
			@show steps_left
			@show time_left
			@show time_per_iter
			@show iterations_left
			@show iter
			if iter == 0
				break
			end
			policy = get_gp_bmdp_policy(bmdp, rng, rollout_depth, iter)
			@show iter
		end

        if isterminal(belief_state) || total_planning_time >= timeout
			if total_planning_time >= timeout
				println("Total steps before timeout: ", length(state_hist))
			end
            break
        end

		#new_belief_state, sim_reward = POMDPs.gen(bmdp, belief_state, a, rng)

		####################################################################
		# This update accesses the true map AFTER the simulations have been completed
		# in the rollouts update_belief() is used which overrides the o from generate_o to only use the GP predicted mean for collecting samples
		s = rand(rng, bmdp.pomdp, belief_state)
		sp, o, r = @gen(:sp, :o, :r)(bmdp.pomdp, s, a, rng)
		new_belief_state = update_belief_no_obs_override(bmdp.pomdp, belief_state, a, o, rng)
		####################################################################

		# just use these to get the true reward NOT the simulated reward
		s = RoverState(belief_state.pos, bmdp.pomdp.true_map, belief_state.cost_expended, belief_state.drill_samples)
		sp = RoverState(new_belief_state.pos, bmdp.pomdp.true_map, new_belief_state.cost_expended, new_belief_state.drill_samples)
		true_reward = reward(bmdp.pomdp, s, a, sp)

        total_reward += true_reward
        belief_state = new_belief_state
		state_hist = vcat(state_hist, deepcopy(belief_state.pos))
		action_hist = vcat(action_hist, deepcopy(a))

        if isterminal(belief_state)
            break
        end
        #path_distance(bmdp.pomdp.ipp_problem, belief_state.pos, bmdp.pomdp.goal_pos)

		# NOTE: we removed gp_hist, reward_hist, total_reward_hist updates here to save memory allocation since they are not currently used
		# gp_hist = vcat(gp_hist, deepcopy(belief_state.location_belief))
		# # location_states_hist = vcat(location_states_hist, deepcopy(state.location_states))
		# reward_hist = vcat(reward_hist, deepcopy(true_reward))
		# total_reward_hist = vcat(total_reward_hist, deepcopy(total_reward))
    end


	y_hist = !bmdp.pomdp.using_AbstractGPs ? belief_state.location_belief.y : belief_state.location_belief.data.δ
    if state_hist[end] != bmdp.pomdp.goal_pos
        position = state_hist[end]
        while position != bmdp.pomdp.goal_pos
			position = shortest_path(all_pairs_shortest_paths, position, bmdp.pomdp.goal_pos)[2]
			action_hist = vcat(action_hist, position) 
            state_hist = vcat(state_hist, position)
			o = bmdp.pomdp.true_map[position] + rand(rng, Normal(0, bmdp.pomdp.σ_max))
			y_hist = vcat(y_hist, o)
        end
    end

	@show length(y_hist)
	@show length(state_hist)
    return total_reward, state_hist, y_hist, gp_hist, action_hist, reward_hist, total_reward_hist, total_planning_time, length(reward_hist)

end


struct GreedyPolicy{P<:POMDPs.POMDP, RNG<:AbstractRNG} <: Policy
    pomdp::P
    rng::RNG
end

function POMDPs.action(p::GreedyPolicy, b::RoverBelief)
    possible_actions = POMDPs.actions(p.pomdp, b)
	possible_rewards = Vector{Float64}()
	for a in possible_actions
		bp = update_belief(p.pomdp, b, a, 0.0, p.rng)
		r = belief_reward(p.pomdp, b, a, bp)
		push!(possible_rewards, r)
	end

    return possible_actions[argmax(possible_rewards)]
end



function get_gp_bmdp_policy(bmdp, rng, max_depth=20, queries = 100)
	rollout_policy = GreedyPolicy(bmdp.pomdp, rng)
	value_estimate = RolloutEstimator(rollout_policy)
	planner = POMDPs.solve(MCTS.DPWSolver(estimate_value=value_estimate, depth=max_depth, n_iterations=queries, rng=rng, k_state=0.5, k_action=10000.0, alpha_state=0.5), bmdp)

	return b -> POMDPs.action(planner, b)
end

# function get_gp_bmdp_policy(bmdp, rng, max_depth=20, queries = 100)
# 	planner = solve(MCTS.DPWSolver(depth=max_depth, n_iterations=queries, rng=rng, k_state=0.5, k_action=10000.0, alpha_state=0.5), bmdp)
# 	# planner = solve(MCTSSolver(depth=max_depth, n_iterations=queries, rng=rng), bmdp)

# 	return b -> action(planner, b)
# end

function random_policy(pomdp, b)
	possible_actions = POMDPs.actions(pomdp, b)
	return rand(pomdp.rng, possible_actions)
end

function mcts_path(ipp_problem, rng, G, N, M, L, Omega, Theta, B, σ_max, σ_min, all_pairs_shortest_paths, run_mcts, run_random, timeout, m_func, number_of_sample_types, rollout_depth, rollout_iterations, objective, true_map::Matrix{Float64}, sample_cost, edge_length, multimodal_sensing::Bool)
	n = isqrt(N)
	Ω = [Omega[i, :] for i in 1:size(Omega, 1)]

	if (objective == "A-IPP" || objective == "D-IPP")
		map_size_sboaippms = (n, n)
		k_sboaippms = with_lengthscale(SqExponentialKernel(), L) # NOTE: check length scale
		X_query_sboaippms = [[Omega[i, 1], Omega[i, 2]] for i in 1:M] #[[i,j] for i = 1:map_size_sboaippms[1], j = 1:map_size_sboaippms[2]]#[[i,j] for i = range(0, 1, length=bins_x+1), j = range(0, 1, length=bins_y+1)]
		# X_query_sboaippms = reshape(X_query_sboaippms, size(X_query_sboaippms)[1]*size(X_query_sboaippms)[2])
		KXqXq_sboaippms = K(X_query_sboaippms, X_query_sboaippms, k_sboaippms)
		GP_sboaippms = GaussianProcess(m_func, μ(X_query_sboaippms, m_func), k_sboaippms, [], X_query_sboaippms, [], [], [], [], [], KXqXq_sboaippms);
		f_prior = GP_sboaippms

		# take a sample at the starting location to be consistent with other TO_AIPPMS methods
		spec_pos = (1,1)
		σ²_n = σ_max^2
		o = true_map[1] + rand(rng, Normal(0, σ_max))
		f_init = posterior(f_prior, [[spec_pos[1], spec_pos[2]]], [o], [σ²_n])
	else
		pos = 1
		# Setup GP
		gp = AbstractGPs.GP(with_lengthscale(SqExponentialKernel(), L))
		# Update GP
		x = Theta[[pos], :]
		X = [x[i, :] for i in 1:size(x, 1)]
		y = true_map[[pos]] + [rand(rng, Normal(0, σ_max))]
		f_init = AbstractGPs.posterior(gp(X, σ_max^2), y)
	end

	#########################################################################################
	pomdp_budget = round(Int, B*(n-1)/edge_length) + 1 # NOTE: this is converting the distance based budget B into a counter budget (counts number of steps taken on the graph as an int to avoid floating point errors)
	pomdp = RoverPOMDP(ipp_problem=ipp_problem, multimodal_sensing=multimodal_sensing, G=G, all_pairs_shortest_paths=all_pairs_shortest_paths, Theta=Theta, true_map=true_map, f_prior=f_init, Ω=Ω, query_size=(M,1), goal_pos=N, cost_budget=pomdp_budget, sample_types= collect(0:(1/number_of_sample_types):(1-1/number_of_sample_types)), σ_max=σ_max, σ_min=σ_min, drill_time = sample_cost, rng=rng, edge_length=edge_length, objective=objective)
	bmdp = BeliefMDP(pomdp, RoverBeliefUpdater(pomdp), belief_reward)
	gp_bmdp_isterminal(s) = POMDPs.isterminal(pomdp, s)
	gp_bmdp_policy = get_gp_bmdp_policy(bmdp, rng, rollout_depth, rollout_iterations)

	# GP-MCTS-DPW
	if run_mcts
		gp_mcts_reward, state_hist, y_hist, gp_hist, action_hist, reward_hist, total_reward_hist, runtime, num_plans = run_rover_bmdp(rng, run_mcts, run_random, bmdp, gp_bmdp_policy, gp_bmdp_isterminal, all_pairs_shortest_paths, timeout, rollout_depth, rollout_iterations)

		path = state_hist
	end

	# RANDOM POLICY
	if run_random
		random_p = b -> random_policy(pomdp, b)
		random_reward, state_hist, y_hist, gp_hist, action_hist, reward_hist, total_reward_hist, runtime, num_plans = run_rover_bmdp(rng, run_mcts, run_random, bmdp, random_p, gp_bmdp_isterminal, all_pairs_shortest_paths, timeout, rollout_depth, rollout_iterations)

		path = state_hist
	end

    if multimodal_sensing
        drills = action_hist .== :drill
		drill_path_idx = findall(x -> x == 1, drills)
		drills = path[drill_path_idx]
        return path, y_hist, drills
    else
        return path, y_hist
    end
end

function solve(ipp_problem::IPP, method::mcts, run_random=false)
    """ 
    Takes in IPP problem definition and returns the path and objective value
    using the solution method specified by method.
    """
    rng = ipp_problem.rng
    G = ipp_problem.Graph.G
    n = ipp_problem.n
    m = ipp_problem.m
    L = ipp_problem.MeasurementModel.L
    Omega = ipp_problem.Graph.Omega
    Theta = ipp_problem.Graph.Theta
    B = ipp_problem.B
    σ_max = ipp_problem.MeasurementModel.σ
    σ_min = 1e-5
    all_pairs_shortest_paths = ipp_problem.Graph.all_pairs_shortest_paths
    run_mcts = run_random ? false : true
    run_random = run_random ? true : false
    timeout = ipp_problem.solution_time
    m_func=(x)->0.0
    number_of_sample_types=10
    rollout_depth=5
    rollout_iterations=100
    obj = ipp_problem.objective
    true_map = ipp_problem.Graph.true_map
    sample_cost = 1
    edge_length = ipp_problem.Graph.edge_length
    multimodal_sensing = false

    path, y_hist = mcts_path(ipp_problem, rng, G, n, m, L, Omega, Theta, B, σ_max, σ_min, all_pairs_shortest_paths, run_mcts, run_random, timeout, m_func, number_of_sample_types, rollout_depth, rollout_iterations, obj, true_map, sample_cost, edge_length, multimodal_sensing)

    if ipp_problem.objective == "expected_improvement"
        return path, adaptive_objective(ipp_problem, path, y_hist), y_hist
    else
        return path, objective(ipp_problem, path)
    end
end

function solve(ipp_problem::IPP, method::random)
    """ 
    Takes in IPP problem definition and returns the path and objective value
    using the solution method specified by method.
    """
    return solve(ipp_problem, mcts(), true)
end

function solve(mmipp::MultimodalIPP, method::mcts, run_random=false)
    """ 
    Takes in MultimodalIPP problem definition and returns the path and objective value
    using the solution method specified by method.
    """
    ipp_problem = mmipp.ipp_problem
    rng = ipp_problem.rng
    G = ipp_problem.Graph.G
    n = ipp_problem.n
    m = ipp_problem.m
    L = ipp_problem.MeasurementModel.L
    Omega = ipp_problem.Graph.Omega
    Theta = ipp_problem.Graph.Theta
    B = ipp_problem.B
    σ_max = mmipp.σ_max
    σ_min = mmipp.σ_min
    all_pairs_shortest_paths = ipp_problem.Graph.all_pairs_shortest_paths
    run_mcts = run_random ? false : true
    run_random = run_random ? true : false
    timeout = ipp_problem.solution_time
    m_func=(x)->0.0
    number_of_sample_types=10
    rollout_depth=5
    rollout_iterations=100
    obj = ipp_problem.objective
    true_map = ipp_problem.Graph.true_map
    sample_cost = 1
    edge_length = ipp_problem.Graph.edge_length
    multimodal_sensing = true

    path, y_hist, drills = mcts_path(ipp_problem, rng, G, n, m, L, Omega, Theta, B, σ_max, σ_min, all_pairs_shortest_paths, run_mcts, run_random, timeout, m_func, number_of_sample_types, rollout_depth, rollout_iterations, obj, true_map, sample_cost, edge_length, multimodal_sensing)

    if mmipp.ipp_problem.objective == "expected_improvement"
        return path, objective(mmipp, path, y_hist, drills), y_hist, drills
    else
        return path, objective(mmipp, path, y_hist, drills), drills
    end
end

function solve(mmipp::MultimodalIPP, method::random)
    """ 
    Takes in IPP problem definition and returns the path and objective value
    using the solution method specified by method.
    """
    return solve(mmipp, mcts(), true)
end