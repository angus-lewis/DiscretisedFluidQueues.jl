module StochasticFluidQueues
import Base: *, size, show, getindex, +, -, setindex!
import Jacobi, LinearAlgebra, SparseArrays
# import Plots, StatsBase, KernelDensity

# model
include("1_SFM.jl")

include("2_abstract_mesh.jl") # things which apply to all meshs
include("3_lazy_generators.jl")
include("4_full_generators.jl")
include("5_SFM_operators.jl")

# auxillary functions
include("6_ME_tools.jl") # used in FRAPApproximation.jl
include("7_polynomials.jl") # used in discontinuous_Galerkin.jl

include("8_discontinuous_Galerkin.jl")
include("9_finite_volume_method.jl")
include("10_FRAP_approximation.jl")

include("11_distributions.jl")

include("12_time_integration.jl")

include("13_simulate.jl")

include("14_plots.jl")



end
