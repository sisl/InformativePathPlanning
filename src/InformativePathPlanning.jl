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
using Pajarito, Hypatia, HiGHS, MosekTools, Gurobi, SCS
import Hypatia.Cones
using Parameters

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

export IPPGraph, 
       MeasurementModel, 
       IPP, 
       MultiagentIPP, 
       MultimodalIPP, 
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
       build_maps

end 