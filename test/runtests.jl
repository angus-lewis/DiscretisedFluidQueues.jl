using DiscretisedFluidQueues 
using LinearAlgebra, SparseArrays
using Test

T = [-2.5 2 0.5; 1 -2 1; 1 2 -3]
T_nz = T - [0.01 0 0;0 0 0;0 0 0]
T_warn = T - [0.000001 0 0;0 0 0;0 0 0]

C = [0.0; 2.0; -3.0]
m = -1 .+ 2*Int.(DiscretisedFluidQueues._strictly_pos.(C))
S = PhaseSet(C)

model = FluidQueue(T,S)

T_aug = [-2.5 0 2 0.5; 0 -2.5 2 0.5; 1 0 -2 1; 0 1 2 -3]
C_aug = [0.0;-0.0;C[2:3]]
m_aug = -1 .+ 2*Int.(DiscretisedFluidQueues._strictly_pos.(C_aug))
S_aug = PhaseSet(C_aug)
am = FluidQueue(T_aug,S_aug)

nodes = [0.0;3.0;4.0;12.0]
nbases = 3
dgmesh = DGMesh(nodes,nbases)

am = augment_model(model)

fv_order = 3
fvmesh = FVMesh(nodes,fv_order)

order = 3
frapmesh = FRAPMesh(nodes,order)

@testset begin

    include("models.jl")

    include("mesh.jl")
    
    include("generators.jl")

# test 5_SFM_operators

    include("me_tools.jl")
    include("polynomials.jl")

    include("distributions.jl")

    include("time_integration.jl")

    include("sims.jl")

    include("numerics.jl")

# more testing for lazy_generators now with new modularised code
# testing for DiscretisedFluidQueue
# etc...
end



