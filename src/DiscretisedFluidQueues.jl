module DiscretisedFluidQueues

import Jacobi, LinearAlgebra, SparseArrays, StaticArrays
import Base: *, +, size, show, getindex, setindex!, sum

# Types
export Model, DiscretisedFluidQueue, BoundedFluidQueue, Phase, PhaseSet # Queues are <:Model
export AbstractMatrixExponential, ConcentratedMatrixExponential, MatrixExponential 
export Mesh, DGMesh, FRAPMesh, FVMesh # are <:Mesh
export Generator, FullGenerator, LazyGenerator # are <:Generator
export SFMDistribution, AbstractIntegrationMethod, AutoQuadrature, Quadrature, TrapezoidRule
export ExplicitRungeKuttaScheme, ForwardEuler, Heuns, StableRK3, StableRK4 # Euler, RungeKutta4 <: TimeIntegrationScheme
export Simulation

# Functions 
export augment_model, membership, N₋, N₊, n_phases, phases, rates # Model methods 
export cell_nodes, Δ, n_bases_per_cell, n_bases_per_phase, n_intervals, total_n_bases # Mesh methods 
export interior_point_mass, left_point_mass, right_point_mass, integrate_time # SFMDistribution methods
export simulate, fixed_time, n_jumps, first_exit_x # Simulation methods
export build_lazy_generator, build_full_generator, static_generator
export cme_params, pdf, ccdf, cdf, build_me, cell_probs
export normalised_closing_operator_cdf, normalised_closing_operator_pdf
export naive_normalised_closing_operator_cdf, naive_normalised_closing_operator_pdf 
export unnormalised_closing_operator_cdf, unnormalised_closing_operator_pdf
export limit, Limiter, NoLimiter, GeneralisedMUSCL

# model
include("SFM.jl")

include("abstract_mesh.jl") # things which apply to all meshes
include("discretised_fluid_queue.jl")
include("lazy_generators.jl")
include("full_generators.jl")

# auxillary functions
include("ME_tools.jl") # used in FRAPApproximation.jl
include("polynomials.jl") # used in discontinuous_Galerkin.jl

include("discontinuous_Galerkin.jl")
include("finite_volume_method.jl")
include("FRAP_approximation.jl")

include("distributions.jl")
include("extend_Base_operations.jl")
include("limiters.jl")

include("time_integration.jl")

include("simulate.jl")

end
