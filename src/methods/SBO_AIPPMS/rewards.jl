function belief_reward(pomdp::RoverPOMDP, b::RoverBelief, a, bp::RoverBelief)
    r = 0.0

    if isterminal(pomdp, b)
        return 0
    else
        if !pomdp.using_AbstractGPs
            if b.location_belief.X == []
                μ_init, ν_init, S_init, EI_init, lcb_init = query_no_data(b.location_belief)
            else
                μ_init, ν_init, S_init, EI_init, lcb_init = query(b.location_belief)
            end

            if bp.location_belief.X == []
                μ_post, ν_post, S_post, EI_post, lcb_post = query_no_data(bp.location_belief)
            else
                μ_post, ν_post, S_post, EI_post, lcb_post = query(bp.location_belief)
            end

            if pomdp.objective == "expected_improvement"
                expected_improvement_reduction = (sum(EI_init) - sum(EI_post))
                r += expected_improvement_reduction
            elseif pomdp.objective == "lower_confidence_bound"
                # for EI and lcb we are not interested in the query points only 
                lcb_reduction = (sum(lcb_init) - sum(lcb_post))
                r += lcb_reduction
            elseif pomdp.objective == "A-IPP"
                # here we look at the variance reduction at the query points (Omega)
                # so we sum over the variance at each query point and we want to maximize the change in variance from init to post
                variance_reduction = (sum(ν_init) - sum(ν_post))
                r += variance_reduction
            elseif pomdp.objective == "D-IPP"
                # here we look at the logdet of the query points (Omega)
                # we want to minimize logdet(Σ) so greater change in logdet(Σ) from init to post is better 
                logdet_reduction = (logdet(S_init) - logdet(S_post))
                r += logdet_reduction
            end
        else
            if pomdp.objective == "expected_improvement"
                # want to go where EI is highest
                query_candidate_point = [pomdp.Theta[bp.pos, :]]
                y_min = minimum(bp.location_belief.data.δ)
    
                σ = sqrt.(var(bp.location_belief(query_candidate_point)))[1]
                μ = mean(bp.location_belief(query_candidate_point))[1]
                EI = expected_improvement_cgp(y_min, μ, σ)
                r += EI
            elseif pomdp.objective == "lower_confidence_bound"
                # want to go where μ - α*σ is lowest, so we want to go to locations that have greater (less negative) -(μ - α*σ)
                query_candidate_point = [pomdp.Theta[bp.pos, :]]
                σ = sqrt.(var(bp.location_belief(query_candidate_point)))
                μ = mean(bp.location_belief(query_candidate_point))
                α = 1.0
                r += -(μ - α*σ)[1]
            elseif pomdp.objective == "A-IPP"
                # here we look at the variance reduction at the query points (Omega)
                # so we sum over the variance at each query point and we want to maximize the change in variance from init to post
                variance_reduction = sum(var(b.location_belief(pomdp.Ω))) - sum(var(bp.location_belief(pomdp.Ω)))
                r += variance_reduction
            elseif pomdp.objective == "D-IPP"
                # here we look at the logdet of the query points (Omega)
                # we want to minimize logdet(Σ) so greater change in logdet(Σ) from init to post is better 
                logdet_reduction = logdet(cov(b.location_belief(Ω))) - logdet(cov(bp.location_belief(Ω))) 
                r += logdet_reduction
            end
        end        
    end

    return r
end

function POMDPs.reward(pomdp::RoverPOMDP, s::RoverState, a, sp::RoverState)
    return 0 # this is called by gen() but is not used
end