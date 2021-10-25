include("src/DiscretisedFluidQueues.jl")
using .DiscretisedFluidQueues
using LinearAlgebra, SparseArrays, StableRNGs
using Test

T = [-2.5 2 0.5; 1 -2 1; 1 2 -3];

C = [0.0; 2.0; -3.0];
m = -1 .+ 2*Int.(DiscretisedFluidQueues._strictly_pos.(C));
S = DiscretisedFluidQueues.PhaseSet(C);

model = DiscretisedFluidQueues.BoundedFluidQueue(T,S);
am = DiscretisedFluidQueues.augment_model(model);
nodes = 0.0:4.0:12.0;
nodes = [0.0;3.0;4.0;12.0]
nbases = 3
dgmesh = DiscretisedFluidQueues.DGMesh(nodes,nbases)

# fv_order = 3
# fvmesh = DiscretisedFluidQueues.FVMesh(nodes,fv_order)

# order = 3;
# frapmesh = DiscretisedFluidQueues.FRAPMesh(nodes,order);

dq = DiscretisedFluidQueues.DiscretisedFluidQueue(am,dgmesh);
B = DiscretisedFluidQueues.build_lazy_generator(dq)
DiscretisedFluidQueues.fast_mul(B,Matrix{Float64}(I(40)))-B
DiscretisedFluidQueues.fast_mul(Matrix{Float64}(I(40)),B)-B