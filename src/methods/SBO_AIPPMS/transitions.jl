function generate_s(pomdp::RoverPOMDP, s::RoverState, a::Any, rng::RNG) where {RNG <: AbstractRNG}

    if isterminal(pomdp, s) #|| isgoal(pomdp, s)
        return RoverState(-1, s.location_states, pomdp.cost_budget, s.drill_samples)
    end

    if a == :drill 
        visit_cost = pomdp.drill_time
    else
        visit_cost = pomdp.visit_cost#compute_visit_cost(pomdp, s.pos, a)#pomdp.dist[s.pos, a]
    end

    if a == :drill
        # new_drill_samples = union(Set{Float64}([s.location_states[s.pos]]), s.drill_samples)
        o = pomdp.true_map[s.pos]#round(rand(rng), digits=4)# observation value does not matter for variance based objective and comparison with MIP
        new_drill_samples = copy(s.drill_samples)#union(Set{Float64}([o]), b.drill_samples)
        push!(new_drill_samples, o)
    else
        new_drill_samples = deepcopy(s.drill_samples)
    end

    new_pos = a == :drill ? s.pos : a
    new_cost_expended = s.cost_expended + visit_cost
    return RoverState(new_pos, s.location_states, new_cost_expended, new_drill_samples)

    # if inbounds(pomdp, new_pos)
    #     new_pos = convert_pos_coord_2_pos_idx(pomdp, new_pos)
    #     new_cost_expended = s.cost_expended + visit_cost

    #     return RoverState(new_pos, s.location_states, new_cost_expended, new_drill_samples)
    # else
    #     return RoverState(s.pos, s.location_states, pomdp.cost_budget*10, new_drill_samples)
    # end
end