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
using Pajarito, Hypatia, HiGHS, Gurobi, MosekTools
using Parameters

abstract type SolutionMethod end
struct ASPC <: SolutionMethod end
struct Greedy <: SolutionMethod end
struct MCTS <: SolutionMethod end
struct Exact <: SolutionMethod end
struct random <: SolutionMethod end
struct DuttaMIP <: SolutionMethod end

@with_kw struct IPPGraph
    G::Vector{Any}                                                          # G[i] returns the neighbors of node i
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
    A::Matrix{Float64}                                                          # meaurement n x m characterization matrix 
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
end

include("utilities/build_graph.jl")
include("utilities/utilities.jl")
include("methods/ASPC.jl")
include("methods/greedy.jl")
include("methods/exact.jl")
include("methods/dutta_mip.jl")
include("utilities/plotting.jl")

function solve(ipp_problem::IPP)
    """ 
    Takes in IPP problem definition and returns the path and objective value.
    """
    return solve(ipp_problem, ASPC())
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
