# Actions
function shortest_path_to_goal(pomdp::RoverPOMDP, pos::Int)
    sp = shortest_path(pomdp.all_pairs_shortest_paths, pos, pomdp.goal_pos)
    return length(sp)
    # pd = path_distance(pomdp.ipp_problem, sp)
    # return round(Int, pd*pomdp.n/pomdp.edge_length)
    # return round(Int, pomdp.path_to_goal_matrix[pos]*pomdp.n/pomdp.edge_length)
end

function actions_possible_from_current(pomdp::RoverPOMDP, pos::Int, cost_expended::Int, drill_samples)
    neighbors_actions = Vector{Any}(pomdp.G[pos])
    # Uncomment line below to allow drilling 
    # push!(neighbors_actions, :drill)
    possible_actions = [] 

    for n in neighbors_actions
        if n == :drill 
            visit_cost = pomdp.drill_time
            return_cost = shortest_path_to_goal(pomdp, pos)
        else
            visit_cost = pomdp.visit_cost#compute_visit_cost(pomdp, pos, n) #pomdp.dist[pos, n]
            return_cost = shortest_path_to_goal(pomdp, n) 
        end

        if n == :drill
            # - Only allow drilling while sum(ν_executed) + sum(ν_left) >= (length(path)-3)*σ_max^2
            #   - Assuming sum(ν_left) is all spectrometer
            # Σν_i >= (d_path -3)σ_max^2

            val1 = (pomdp.σ_min*(length(drill_samples)+1) + pomdp.σ_max^2 * (cost_expended - pomdp.drill_time*length(drill_samples)+1) + pomdp.σ_max^2 * return_cost) 
            val2 = ((pomdp.cost_budget - 3)*pomdp.σ_max^2)
            
            if val1 > val2
            # if (pomdp.σ_min*(length(drill_samples)+1) + pomdp.σ_max^2 * (cost_expended_scaled - pomdp.drill_time*length(drill_samples)+1) + pomdp.σ_max^2 * return_cost_scaled) > ((pomdp.cost_budget - 3)*pomdp.σ_max^2)*sqrt(pomdp.goal_pos)
                # don't allow any more drilling if this constraint is violated
                visit_cost = Inf
            end
        end

        if (cost_expended + visit_cost + return_cost) <= pomdp.cost_budget
            if n == :drill && length(drill_samples) >= 3
                # skip
            else
                push!(possible_actions, n)
            end
        end

    end

    return (possible_actions...,) # return as a tuple
end


function POMDPs.actions(pomdp::RoverPOMDP, s::RoverState)
    if isterminal(pomdp, s)
        # return (:up, :down, :left, :right, :wait, :NE, :NW, :SE, :SW, :drill)
        return (:up)

    else
        possible_actions = actions_possible_from_current(pomdp, s.pos, s.cost_expended, s.drill_samples)
    end
    return possible_actions
end

# function POMDPs.action(b::RoverBelief)
#     if isterminal(pomdp, b)
#         return (:up, :down, :left, :right, :wait, :NE, :NW, :SE, :SW, :drill)
#     else
#         possible_actions = actions_possible_from_current(pomdp, b.pos, b.cost_expended)
#     end
#     return possible_actions
# end
#
function POMDPs.action(p::RandomPolicy, b::RoverBelief)
    possible_actions = POMDPs.actions(p.problem, b)
    return rand(p.problem.pomdp.rng, possible_actions)
    # return possible_actions[rand(collect(1:length(possible_actions)))]
end

function POMDPs.actions(pomdp::RoverPOMDP, s::LeafNodeBelief)
    s = s.sp
    if isterminal(pomdp, s)
        # return (:up, :down, :left, :right, :wait, :NE, :NW, :SE, :SW, :drill)
        return (:up)
    else
        possible_actions = actions_possible_from_current(pomdp, s.pos, s.cost_expended, s.drill_samples)
    end
    return possible_actions
end

function POMDPs.actions(pomdp::RoverPOMDP, b::RoverBelief)
    if isterminal(pomdp, b)
        # return (:up, :down, :left, :right, :wait, :NE, :NW, :SE, :SW, :drill)
        return (:up)
    else
        possible_actions = actions_possible_from_current(pomdp, b.pos, b.cost_expended, b.drill_samples)
    end
    return possible_actions
end

# const dir = Dict(:up=>RoverPos(0,1), :down=>RoverPos(0,-1), :left=>RoverPos(-1,0), :right=>RoverPos(1,0), :wait=>RoverPos(0,0), :NE=>RoverPos(1,1), :NW=>RoverPos(-1,1), :SE=>RoverPos(1,-1), :SW=>RoverPos(-1,-1), :drill=>RoverPos(0,0))
# const aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4, :wait=>5, :NE=>6, :NW=>7, :SE=>8, :SW=>9, :drill=>10)
# const dir = Dict(:up=>RoverPos(0,1), :down=>RoverPos(0,-1), :left=>RoverPos(-1,0), :right=>RoverPos(1,0), :drill=>RoverPos(0,0))
# const aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4, :drill=>10)


# POMDPs.actionindex(POMDP::RoverPOMDP, a::Symbol) = aind[a]
POMDPs.actionindex(POMDP::RoverPOMDP, a) = a == :drill ? N+1 : a