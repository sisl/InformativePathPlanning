using POMDPModels, POMDPs, POMDPPolicies
using StaticArrays, Parameters, Random, POMDPModelTools, Distributions
using Images, LinearAlgebra, Printf
using Plots
using StatsBase
using BasicPOMCP


export
    RoverPOMDP,
    RoverState,
    RoverBelief

struct RoverState
    pos::Int
    location_states::Matrix{Float64}
    cost_expended::Int
    drill_samples::Vector{Float64}
end

struct RoverBelief
    pos::Int
    location_belief #GaussianProcess or AbstractGPs.PosteriorGP
    cost_expended::Int
    drill_samples::Vector{Float64}
end


@with_kw mutable struct RoverPOMDP <: POMDP{RoverState, Any, Float64} # POMDP{State, Action, Observation}
    ipp_problem::IPP
    multimodal_sensing::Bool               = false
    G::Vector{Vector{Int64}}
    all_pairs_shortest_paths::Graphs.FloydWarshallState{Float64, Int64}
    Theta::Matrix{Float64}
    true_map::Matrix{Float64}
    f_prior
    Ω::Vector{Vector{Float64}}
    init_pos::Int                          = 1
    query_size::Tuple{Int, Int}            = (101,51)
    goal_pos::Int                          
    n::Int                                 = round(Int, sqrt(goal_pos))
    visit_cost::Int                        = 1
    edge_length::Int

    tprob::Float64                         = 1
    discount::Float64                      = 1.0
    σ_min::Float64                         = σ_min
    σ_max::Float64                         = σ_max
    drill_time::Int                        = sample_cost #3.0 # takes 10 time units to drill vs. 1.0 to move to neighbor cell
    cost_budget::Int                       = total_budget#300.0
    sample_types::Vector{Float64}          = collect(0:0.1:0.9)
    rng::AbstractRNG
    objective::String
    using_AbstractGPs::Bool                = (objective == "expected_improvement" || objective == "lower_confidence_bound") ? true : false
end

function POMDPs.isterminal(pomdp::RoverPOMDP, s::RoverState)
    if s.cost_expended + shortest_path_to_goal(pomdp, s.pos) >= pomdp.cost_budget
        return true

    elseif s.pos == pomdp.goal_pos

        neighbor_actions = actions_possible_from_current(pomdp, s.pos, s.cost_expended, s.drill_samples)

        if neighbor_actions == ()
            return true
        else
            min_cost_to_goal = minimum([shortest_path_to_goal(pomdp, n) for n in neighbor_actions if n != :drill])
        end

        # min_cost_from_start = minimum([pomdp.shortest_paths[init_idx, n] for n in neighbors])
        if (pomdp.cost_budget - s.cost_expended) < 2*min_cost_to_goal
            return true
        else
            return false
        end
    else
        return false
    end

end

function POMDPs.isterminal(pomdp::RoverPOMDP, b::RoverBelief)
    return isterminal(pomdp, RoverState(b.pos, zeros(pomdp.query_size), b.cost_expended, b.drill_samples))
    # return isterminal(pomdp, rand(pomdp.rng, pomdp, b))
end

function POMDPs.gen(pomdp::RoverPOMDP, s::RoverState, a::Any, rng::RNG) where {RNG <: AbstractRNG}
    sp = generate_s(pomdp, s, a, rng)
    o = generate_o(pomdp, s, a, sp, rng)
    r = reward(pomdp, s, a, sp)

    return (sp=sp, o=o, r=r)
end

function compute_visit_cost(pomdp, pos1, pos2)
    return norm(pomdp.Theta[pos1, :] - pomdp.Theta[pos2, :])
end

# discount
POMDPs.discount(pomdp::RoverPOMDP) = pomdp.discount


include("states.jl")
include("actions.jl")
include("observations.jl")
include("beliefs.jl")
include("transitions.jl")
include("rewards.jl")
