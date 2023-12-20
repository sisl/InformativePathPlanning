# Informative Path Planning with Dynamic Programming

We consider the problem of finding an informative path through a graph, given initial and terminal nodes and a given maximum path length. We assume that a linear noise corrupted measurement is taken at each node of an underlying unknown vector that we wish to estimate. The informativeness is measured by the reduction in uncertainty in our estimate, evaluated using several metrics. We present a convex relaxation for this informative path planning problem, which we can readily solve to obtain a bound on the possible performance. We develop a model predictive control (MPC)-type method where we construct the path segment by segment through dynamic programming. This involves solving an orienteering problem, with the node reward acting as a surrogate for informativeness, taking the first step, and then repeating the process. The method scales to very large problem instances and achieves performance not too far from the bound produced by the convex relaxation.

# Instructions

Clone the repository: 
```
git clone https://github.com/josh0tt/InformativePathPlanning.git
```

Navigate to the project directory and add data and figure folders:
```
cd /path/to/InformativePathPlanning
mkdir data
mkdir data/graph_cache
mkdir data/maps
mkdir figures
```

Activate the project:
```
julia> ]
(v1.x) pkg> activate /path/to/InformativePathPlanning
```

To run the simple example:
```
julia> include("simple_example.jl")
```

To reproduce the figures from the paper:
```
julia> include("run_paper_experiments.jl")
julia> figure_1(load_data=false)
```