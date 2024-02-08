# function convert_pos_idx_2_pos_coord(pomdp::RoverPOMDP, pos::Int)
#     if pos == -1
#         return RoverPos(-1,-1)
#     else
#         return RoverPos(CartesianIndices(pomdp.map_size)[pos].I[1], CartesianIndices(pomdp.map_size)[pos].I[2])
#     end
# end

# function convert_pos_coord_2_pos_idx(pomdp::RoverPOMDP, pos::RoverPos)
#     if pos == RoverPos(-1,-1)
#         return -1
#     else
#         return LinearIndices(pomdp.map_size)[pos[1], pos[2]]
#     end
# end

function POMDPs.initialstate(pomdp::RoverPOMDP)
    return RoverState(pomdp.init_pos, pomdp.true_map, 0.0, Vector{Float64}(Float64[]))
end