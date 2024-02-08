# Approximate Sequential Optimization for Informative Path Planning

We consider the problem of finding an informative path through a graph, given initial and terminal nodes and a given maximum path length. We assume that a linear noise corrupted measurement is taken at each node of an underlying unknown vector that we wish to estimate. The informativeness is measured by the reduction in uncertainty in our estimate, evaluated using several metrics. We present a convex relaxation for this informative path planning problem, which we can readily solve to obtain a bound on the possible performance. We develop an approximate sequential method where the path is constructed segment by segment through dynamic programming. This involves solving an orienteering problem, with the node reward acting as a surrogate for informativeness, taking the first step, and then repeating the process. The method scales to very large problem instances and achieves performance not too far from the bound produced by the convex relaxation. We also demonstrate our method's ability to handle adaptive objectives, multimodal sensing, and multiagent variations of the informative path planning problem.

# Instructions

Use the julia package manager to add the InformativePathPlanning module:: 
```julia 
] add https://github.com/josh0tt/InformativePathPlanning.git
using InformativePathPlanning
```

To run the simple example:
```julia
julia> run_simple_example()
```

To run the multiagent example:
```julia
julia> run_multiagent_example()
```

To reproduce the figures from the paper:
```julia
julia> run_paper_experiments()
```
Note that the `data` and `figures` directories are included by default so that you do not have to create them before running the paper experiments. 

# Solver types
If you don't have access to Mosek and Gurobi licenses, you can specify the solver type as `"open"` inside of the IPP struct. By default, it is set to `"commercial"`.   

# Examples
Trajectories generated from each of the methods and baselines with `n=625` graph nodes and `m=20` prediction locations. This example also includes multimodal sensor selection where the yellow points indicate more accurate sensing measurements. 
<p align="center">
  <img alt="Variance" src="https://github.com/sisl/InformativePathPlanning/blob/main/img/trajectories.png" width="100%">
</p>

Multiagent with obstacles

<p align="center">
  <img alt="Variance" src="https://github.com/sisl/InformativePathPlanning/blob/main/img/multiagent.gif" width="100%">
</p>