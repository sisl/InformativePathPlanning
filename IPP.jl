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
struct MCTS <: SolutionMethod end
struct ASPC <: SolutionMethod end
struct Exact <: SolutionMethod end
struct Greedy <: SolutionMethod end
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
    edge_length::Int                                                        # length of one side of the grid (assumes square grid with dimensions edge_length x edge_length)
end

@with_kw struct MeasurementModel
    σ::Float64                                                              # measurement standard deviation
    Σₓ::Matrix{Float64}                                                     # prior covariance matrix 
    L::Float64                                                              # length scale used in the kernel to build the covariance matrix
    A::Matrix{Int}                                                          # meaurement n x m characterization matrix 
end

@with_kw struct IPP
    rng::MersenneTwister
    n::Int                                                                  # number of nodes
    m::Int                                                                  # number of target prediction locations
    Graph::IPPGraph
    MeasurementModel::MeasurementModel
    objective::String                                                       # A-IPP, D-IPP, expected_improvement
    B::Int                                                                  # budget
    true_map::Matrix{Float64}                                               # map of true values at all nodes in the environment
    solution_time::Float64                                                  # time allowed to find a solution
    replan_rate::Int                                                       # replan after replan_rate steps 
end

include("build_graph.jl")
include("utilities.jl")
include("ASPC.jl")
include("plotting.jl")

function solve(ipp_problem::IPP)
    """ 
    Takes in IPP problem definition and returns the path and objective value.
    """
    return solve(ipp_problem, ASPC())
end

function solve(ipp_problem::IPP, method::ASPC)
    """ 
    Takes in IPP problem definition and returns the path and objective value
    using the solution method specified by method.
    """

    path = Vector{Int64}([ipp_problem.Graph.start])
    gp, y_hist = initialize_gp(ipp_problem)
    time_left = ipp_problem.solution_time

    while path[end] != ipp_problem.Graph.goal && time_left > 0
        planned_path, planning_time = @timed action(ipp_problem, method, gp, path)
        time_left -= planning_time

        push!(path, planned_path[2:(2+ipp_problem.replan_rate-1)]...)
        gp, y_hist = update_gp(ipp_problem, gp, y_hist, planned_path[2:(2+ipp_problem.replan_rate-1)])

        if length(planned_path[(2+ipp_problem.replan_rate):end]) <= ipp_problem.replan_rate
            push!(path, planned_path[(2+ipp_problem.replan_rate):end]...)
            gp, y_hist = update_gp(ipp_problem, gp, y_hist, planned_path[(2+ipp_problem.replan_rate):end])
            break
        end
    end

    return path, objective(ipp_problem, path, y_hist)
end