using DiscretisedFluidQueues 
using LinearAlgebra, SparseArrays, StableRNGs
using Test

T = [-2.5 2 0.5; 1 -2 1; 1 2 -3]
T_nz = T - [0.01 0 0;0 0 0;0 0 0]
T_warn = T - [0.000001 0 0;0 0 0;0 0 0]

C = [0.0; 2.0; -3.0]
m = -1 .+ 2*Int.(DiscretisedFluidQueues._strictly_pos.(C))
S = PhaseSet(C)

model = BoundedFluidQueue(T,S)

nodes = collect(0.0:4:12.0)
nodes = [0.0;3.0;4.0;12.0]
nbases = 3
dgmesh = DGMesh(nodes,nbases)

am = augment_model(model)
fv_order = 3
fvmesh = FVMesh(nodes,fv_order)

order = 3
frapmesh = FRAPMesh(nodes,order)

i = :dgmesh

dq = @eval DiscretisedFluidQueue(am,$i)
B = build_lazy_generator(dq)
