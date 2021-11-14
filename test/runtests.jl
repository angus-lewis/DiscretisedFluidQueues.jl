using DiscretisedFluidQueues 
using LinearAlgebra, SparseArrays, StableRNGs 
using Test
include("SFM_operators.jl")

T = [-2.5 2 0.5; 1 -2 1; 1 2 -3]
T_nz = T - [0.01 0 0;0 0 0;0 0 0]
T_warn = T - [0.000001 0 0;0 0 0;0 0 0]

C = [0.0; 2.0; -3.0]
m = -1 .+ 2*Int.(DiscretisedFluidQueues._strictly_pos.(C))
S = PhaseSet(C)

model = BoundedFluidQueue(T,S)

T_aug = [-2.5 0 2 0.5; 0 -2.5 2 0.5; 1 0 -2 1; 0 1 2 -3]
C_aug = [0.0;-0.0;C[2:3]]
m_aug = -1 .+ 2*Int.(DiscretisedFluidQueues._strictly_pos.(C_aug))
S_aug = PhaseSet(C_aug)
am = BoundedFluidQueue(T_aug,S_aug)

nds_vec = Array{Any,1}(undef,2)
nds_vec[1] = [0.0;3.0;4.0;12.0]
nds_vec[2] = 0.0:4.0:12.0
for nds in 1:2 # probably overkill to test everything twice but oh well!
    global nodes=nds_vec[nds]

    global nbases=3
    global dgmesh=DGMesh(nodes,nbases)

    global am=augment_model(model)

    global fv_order=3
    global fvmesh=FVMesh(nodes,fv_order)

    global order=3
    global frapmesh=FRAPMesh(nodes,order)

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

        include("limiters.jl")

    # more testing for lazy_generators now with new modularised code
    # testing for DiscretisedFluidQueue
    # etc...
    end
end