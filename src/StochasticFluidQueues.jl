module StochasticFluidQueues
import Base: *, size, show, getindex, +, -, setindex!
import Jacobi, LinearAlgebra, SparseArrays
import Plots, StatsBase, KernelDensity

# model
include("SFM.jl")
include("SFM_operators.jl")

# auxillary functions
include("ME_tools.jl") # used in FRAPApproximation.jl
include("polynomials.jl") # used in discontinuous_Galerkin.jl

include("abstract_mesh.jl") # things which apply to all meshs

include("discontinuous_Galerkin.jl")
include("FVM.jl")
include("FRAPApproximation.jl")

include("Distributions.jl")

include("time_integration.jl")

include("SimulateSFFM.jl")

include("Plots.jl")

# export get_rates, n_phases, phases, N₋, N₊, Model, PhaseSet, FluidQueue, augment_model

end
