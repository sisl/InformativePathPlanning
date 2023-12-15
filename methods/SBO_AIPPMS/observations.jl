# Observations

function generate_o(pomdp::RoverPOMDP, s::RoverState, action::Any, sp::RoverState, rng::AbstractRNG)
    if isterminal(pomdp, sp)
        # NOTE: we can't return -1.0 here because this will mess up expected improvement calculations
        #return -1.0 
        return pomdp.true_map[sp.pos] + rand(rng, Normal(0, pomdp.σ_max))
    end

    # Remember you make the observation at sp NOT s
    if action == :drill
        o = pomdp.true_map[sp.pos]#sp.location_states[sp.pos]
    else
        # NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
        # for normal dist whereas our GP setup uses σ²_n

        o = pomdp.true_map[sp.pos] + rand(rng, Normal(0, pomdp.σ_max))
    end

    return o
end