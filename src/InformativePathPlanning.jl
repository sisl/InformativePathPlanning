module InformativePathPlanning

using JuMP
using Random
using LinearAlgebra
using Combinatorics
using Plots
using Graphs, SimpleWeightedGraphs, SparseArrays, GraphRecipes
using JLD2
using SharedArrays
using ProgressMeter
using StatsBase
using KernelFunctions
using ScatteredInterpolation
using Distances
using Distributions
using AbstractGPs
using TickTock
using Pajarito, Hypatia, HiGHS, SCS 
import Hypatia.Cones
using Parameters
using Requires
using POMDPs
using BasicPOMCP

function __init__()
    @require Gurobi="2e9cd046-0924-5485-92f1-d5272153d98b" begin
        @eval using .Gurobi
        global const gurobi_available = true
        println("Gurobi is available.")
    end

    @require MosekTools="1ec41992-ff65-5c91-ac43-2df89e9693a4" begin
        @eval using .MosekTools
        global const mosek_available = true
        println("Mosek is available.")
    end

end

abstract type SolutionMethod end
struct ASPO <: SolutionMethod end
struct Greedy <: SolutionMethod end
struct mcts <: SolutionMethod end
struct Exact <: SolutionMethod end
struct trΣ⁻¹ <: SolutionMethod end
struct random <: SolutionMethod end
struct DuttaMIP <: SolutionMethod end

@with_kw struct IPPGraph
    G::Vector{Vector{Int64}}                                                # G[i] returns the neighbors of node i
    start::Int                                                              # start node
    goal::Int                                                               # goal node
    Theta::Matrix{Float64}                                                  # location of the graph nodes
    Omega::Matrix{Float64}                                                  # location of target prediction nodes
    all_pairs_shortest_paths::Graphs.FloydWarshallState{Float64, Int64}     # all shortest paths from node i to node j precomputed for the graph
    distances::Matrix{Float64}                                              # distances between all nodes in the graph
    true_map::Matrix{Float64}                                               # map of true values at all nodes in the environment
    edge_length::Int                                                        # length of one side of the grid (assumes square grid with dimensions edge_length x edge_length)
end

@with_kw struct MeasurementModel
    σ::Float64                                                              # measurement standard deviation
    Σₓ::Matrix{Float64}                                                     # prior covariance matrix
    Σₓ⁻¹::Matrix{Float64}                                                   # inverse of prior covariance matrix 
    L::Float64                                                              # length scale used in the kernel to build the covariance matrix
    A::Matrix{Float64}                                                      # meaurement n x m characterization matrix 
end

@with_kw struct IPP
    rng::MersenneTwister
    n::Int                                                                  # number of nodes
    m::Int                                                                  # number of target prediction locations
    Graph::IPPGraph
    MeasurementModel::MeasurementModel
    objective::String                                                       # A-IPP, D-IPP, expected_improvement
    B::Int                                                                  # budget
    solution_time::Float64                                                  # time allowed to find a solution
    replan_rate::Int                                                        # replan after replan_rate steps
    solver_type::String = "commercial"                                    # open or commercial solvers
end

# Providing a constructor with default value for solver_type
function IPP(rng::MersenneTwister, n::Int, m::Int, Graph::IPPGraph, MeasurementModel::MeasurementModel, objective::String, B::Int, solution_time::Float64, replan_rate::Int)
    return IPP(rng, n, m, Graph, MeasurementModel, objective, B, solution_time, replan_rate, "commercial")
end

@with_kw struct MultiagentIPP
    ipp_problem::IPP                                                        # IPP problem
    M::Int                                                                  # Number of agents
end

@with_kw struct MultimodalIPP
    ipp_problem::IPP                                                        # IPP problem
    σ_min::Float64                                                          # highest quality sensor
    σ_max::Float64                                                          # lowest quality sensor
    k::Int                                                                  # number of sensor types 
end

struct PointsOfInterestIPP
    ipp_problem::IPP                                                        # IPP problem
    noi::Vector{Int64}                                                      # Nodes of interest in the IPP problem graph
end

@with_kw struct MultimodalPOIIPP
    poiipp::PointsOfInterestIPP                                             # PointsOfInterestIPP problem
    σ_min::Float64                                                          # highest quality sensor
    σ_max::Float64                                                          # lowest quality sensor
    k::Int                                                                  # number of sensor types 
end

include("utilities/build_graph.jl")
include("utilities/utilities.jl")
include("methods/aspo.jl")
include("methods/greedy.jl")
include("methods/exact.jl")
include("methods/dutta_mip.jl")
include("methods/mcts.jl")
include("utilities/plotting.jl")
include("multimodal_sensor_selection.jl")
include("utilities/build_maps.jl")
include("methods/points_of_interest.jl")

function solve(ipp_problem::MultiagentIPP)
    """ 
    Takes in MultiagentIPP problem definition and returns M paths and the objective value.
    """
    return solve(ipp_problem, ASPO())
end

function solve(ipp_problem::MultimodalIPP)
    """ 
    Takes in MultimodalIPP problem definition and returns path and the objective value.
    """
    return solve(ipp_problem, ASPO())
end

function solve(mmipp::MultimodalIPP, method::SolutionMethod)
    """ 
    Takes in MultimodalIPP problem definition and returns the path and objective value
    using the solution method specified by method.
    """

    path, objVal = solve(mmipp.ipp_problem, method)

    if mmipp.ipp_problem.objective ∉ ["A-IPP", "D-IPP"]
        @error("You tried to run multimodal sensing for an objective that is not implemented")
    else
        drills, drill_time = @timed (mmipp.ipp_problem.objective == "A-IPP" ? run_a_optimal_sensor_selection(mmipp, unique(path)) : run_d_optimal_sensor_selection(mmipp, unique(path)))
    end

    return path, drills, objVal
end

function solve(mmipp::MultimodalPOIIPP, method::SolutionMethod)
    """ 
    Takes in MultimodalPOIIPP problem definition and returns the path and objective value
    using the solution method specified by method.
    """

    if typeof(method) != Exact && typeof(method) != trΣ⁻¹
        @error("You tried to run multimodal sensing for a method that is not implemented")
    else
        path, objVal = solve(mmipp.poiipp, method)

        if mmipp.poiipp.ipp_problem.objective ∉ ["A-IPP", "D-IPP"]
            @error("You tried to run multimodal sensing for an objective that is not implemented")
        else
            drills, drill_time = @timed (mmipp.poiipp.ipp_problem.objective == "A-IPP" ? run_a_optimal_sensor_selection(mmipp, unique(path)) : run_d_optimal_sensor_selection(mmipp, unique(path)))
        end

        return path, drills, objVal
    end
end

function solve(ipp_problem::IPP)
    """ 
    Takes in IPP problem definition and returns the path and objective value.
    """
    return solve(ipp_problem, ASPO())
end

function relax(ipp_problem::IPP)
    """ 
    Takes in IPP problem definition and returns the lower bound objective value for A and D-IPP.
    """
    if ipp_problem.objective ∉ ["A-IPP", "D-IPP"]
        @error("You tried to return a lower bound for an objective that is not implemented")
    else
        _, lower_bound = solve(ipp_problem, Exact(), true)
        return lower_bound
    end
end

include("simple_example.jl")
include("multiagent_example.jl")
include("multimodal_example.jl")
include("poi_example.jl")

export IPPGraph, 
    MeasurementModel, 
    IPP, 
    MultiagentIPP, 
    MultimodalIPP,
    MultimodalPOIIPP,
    PointsOfInterestIPP, 
    ASPO, 
    Greedy, 
    mcts, 
    Exact, 
    trΣ⁻¹, 
    random, 
    DuttaMIP, 
    solve, 
    relax, 
    run_simple_example, 
    build_graph, 
    run_multiagent_example, 
    run_multimodal_example,
    run_poi_example,
    run_poi_multimodal_example,
    build_maps,
    kernel,
    figure_1,
    figure_2,
    figure_3,
    figure_4,
    figure_5,
    figure_6,
    figure_7,
    figure_9,
    gurobi_available,
    mosek_available

end 