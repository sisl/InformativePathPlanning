struct RoverBeliefUpdater{P<:POMDPs.POMDP} <: Updater
    pomdp::P
end

function Base.rand(rng::AbstractRNG, pomdp::RoverPOMDP, b::RoverBelief)

    if !pomdp.using_AbstractGPs
        location_states = rand(rng, b.location_belief, b.location_belief.mXq, b.location_belief.KXqXq)
    else
        location_states = rand(rng, b.location_belief(pomdp.Ω))
    end
    location_states = reshape(location_states, pomdp.query_size)

    return RoverState(b.pos, location_states, b.cost_expended, b.drill_samples)
end


function POMDPs.update(updater::RoverBeliefUpdater, b::RoverBelief, a::Any, o::Float64)
    ub = update_belief(updater.pomdp, b, a, o, updater.pomdp.rng)
    return ub
end


function update_belief(pomdp::P, b::RoverBelief, a::Any, o::Float64, rng::RNG) where {P <: POMDPs.POMDP, RNG <: AbstractRNG}
    if isterminal(pomdp, b)
        return b
    end

    if a == :drill 
        visit_cost = pomdp.drill_time
    else
        visit_cost = pomdp.visit_cost#compute_visit_cost(pomdp, b.pos, a)#pomdp.dist[b.pos, a]
    end

    ####################################################################
    # override observation so we are not accessing the true map
    ####################################################################
    new_pos = a == :drill ? b.pos : a
    if !pomdp.using_AbstractGPs
        μₚ, νₚ, S, EI, lcb = b.location_belief.X == [] ? query_no_data(b.location_belief) : query(b.location_belief)
        o = a == :drill ? pomdp.true_map[b.pos] :  pomdp.true_map[a] + rand(rng, Normal(0, pomdp.σ_max)) # doesn't matter that we access true map here since it is for A and D optimal objectives
        # o = a == :drill ? μₚ[b.pos] : rand(rng, Normal(μₚ[new_pos], pomdp.σ_max))
    else
        # use gp mean to predict value of candidate point
        o = mean(b.location_belief([pomdp.Theta[new_pos, :]]))[1]
    end 
    
    ####################################################################

    # NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
    # for normal dist whereas our GP setup uses σ²_n

    if a == :drill
        # if we drill we fully collapse the belief
        drill_pos = pomdp.Theta[b.pos, :] #convert_pos_idx_2_pos_coord(pomdp, b.pos)

        if !pomdp.using_AbstractGPs
            σ²_n = pomdp.σ_min #^2 dont square this causes singular exception in GP update
            f_posterior = posterior(b.location_belief, [[drill_pos[1], drill_pos[2]]], [o], [σ²_n])
        else
            x = pomdp.Theta[[b.pos], :]
            X = [x[i, :] for i in 1:size(x, 1)]
            f_posterior = AbstractGPs.posterior(b.location_belief(X, pomdp.σ_min), [o])
        end

        new_cost_expended = b.cost_expended + visit_cost
        new_drill_samples = copy(b.drill_samples)#union(Set{Float64}([o]), b.drill_samples)
        push!(new_drill_samples, o)

        return RoverBelief(b.pos, f_posterior, new_cost_expended, new_drill_samples)

    else
        new_pos = a#convert_pos_coord_2_pos_idx(pomdp, pos)
        new_cost_expended = b.cost_expended + visit_cost

        spec_pos = pomdp.Theta[new_pos, :]#convert_pos_idx_2_pos_coord(pomdp, new_pos)
        
        if !pomdp.using_AbstractGPs
            σ²_n = pomdp.σ_max^2
            f_posterior = posterior(b.location_belief, [[spec_pos[1], spec_pos[2]]], [o], [σ²_n])
        else
            x = pomdp.Theta[[new_pos], :]
            X = [x[i, :] for i in 1:size(x, 1)]
            f_posterior = AbstractGPs.posterior(b.location_belief(X, pomdp.σ_max), [o])
        end

        return RoverBelief(new_pos, f_posterior, new_cost_expended, b.drill_samples)

    end
end

function update_belief_no_obs_override(pomdp::P, b::RoverBelief, a::Any, o::Float64, rng::RNG) where {P <: POMDPs.POMDP, RNG <: AbstractRNG}
    if isterminal(pomdp, b)
        return b
    end

    if a == :drill 
        visit_cost = pomdp.drill_time
    else
        visit_cost = pomdp.visit_cost#compute_visit_cost(pomdp, b.pos, a)#pomdp.dist[b.pos, a]
    end

    # NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
    # for normal dist whereas our GP setup uses σ²_n

    if a == :drill
        # if we drill we fully collapse the belief
        drill_pos = pomdp.Theta[b.pos, :] #convert_pos_idx_2_pos_coord(pomdp, b.pos)

        if !pomdp.using_AbstractGPs
            σ²_n = pomdp.σ_min #^2 dont square this causes singular exception in GP update
            f_posterior = posterior(b.location_belief, [[drill_pos[1], drill_pos[2]]], [o], [σ²_n])
        else
            x = pomdp.Theta[[b.pos], :]
            X = [x[i, :] for i in 1:size(x, 1)]
            f_posterior = AbstractGPs.posterior(b.location_belief(X, pomdp.σ_min), [o])
        end

        new_cost_expended = b.cost_expended + visit_cost
        new_drill_samples = copy(b.drill_samples)#union(Set{Float64}([o]), b.drill_samples)
        push!(new_drill_samples, o)

        return RoverBelief(b.pos, f_posterior, new_cost_expended, new_drill_samples)

    else
        new_pos = a#convert_pos_coord_2_pos_idx(pomdp, pos)
        new_cost_expended = b.cost_expended + visit_cost

        spec_pos = pomdp.Theta[new_pos, :]#convert_pos_idx_2_pos_coord(pomdp, new_pos)
        
        if !pomdp.using_AbstractGPs
            σ²_n = pomdp.σ_max^2
            f_posterior = posterior(b.location_belief, [[spec_pos[1], spec_pos[2]]], [o], [σ²_n])
        else
            x = pomdp.Theta[[new_pos], :]
            X = [x[i, :] for i in 1:size(x, 1)]
            f_posterior = AbstractGPs.posterior(b.location_belief(X, pomdp.σ_max), [o])
        end

        return RoverBelief(new_pos, f_posterior, new_cost_expended, b.drill_samples)

    end
end


function BasicPOMCP.extract_belief(::RoverBeliefUpdater, node::BeliefNode)
    return node
end


function POMDPs.initialize_belief(updater::RoverBeliefUpdater, d)
    return initial_belief_state(updater.pomdp, updater.pomdp.rng)
end


function POMDPs.initialize_belief(updater::RoverBeliefUpdater, d, rng::RNG) where {RNG <: AbstractRNG}
    return initial_belief_state(updater.pomdp, rng)
end

function initial_belief_state(pomdp::RoverPOMDP, rng::RNG) where {RNG <: AbstractRNG}

    pos = pomdp.init_pos
    location_belief = pomdp.f_prior
    cost_expended = 0.0
    drill_samples = Vector{Float64}(Float64[])#Set{Float64}(Float64[])

    return RoverBelief(pos, location_belief, cost_expended, drill_samples)

end