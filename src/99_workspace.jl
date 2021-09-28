# using LinearAlgebra, SparseArrays #Revise
# include("src/DiscretisedFluidQueues.jl")
# import .DiscretisedFluidQueues 
# using Test

T = [-2.5 2 0.5; 1 -2 1; 1 2 -3]
T_nz = T - [0.01 0 0;0 0 0;0 0 0]
T_warn = T - [0.000001 0 0;0 0 0;0 0 0]

C = [0.0; 2.0; -3.0]
m = -1 .+ 2*Int.(DiscretisedFluidQueues._strictly_pos.(C))
S = DiscretisedFluidQueues.PhaseSet(C)

model = DiscretisedFluidQueues.FluidQueue(T,S)

T_aug = [-2.5 0 2 0.5; 0 -2.5 2 0.5; 1 0 -2 1; 0 1 2 -3]
C_aug = [0.0;-0.0;C[2:3]]
m_aug = -1 .+ 2*Int.(DiscretisedFluidQueues._strictly_pos.(C_aug))
S_aug = DiscretisedFluidQueues.PhaseSet(C_aug)
am = DiscretisedFluidQueues.FluidQueue(T_aug,S_aug)

nodes = collect(0:0.5:10)#[0.0;3.0;4.0;12.0]
nbases = 15
dgmesh = DiscretisedFluidQueues.DGMesh(nodes,nbases)

am = DiscretisedFluidQueues.augment_model(model)

fv_order = 15
fvmesh = DiscretisedFluidQueues.FVMesh(nodes,fv_order)

order = 15
frapmesh = DiscretisedFluidQueues.FRAPMesh(nodes,order)

dq = DiscretisedFluidQueues.DiscretisedFluidQueue(model,frapmesh)

lz = DiscretisedFluidQueues.build_lazy_generator(dq)

@macroexpand DiscretisedFluidQueues.@static_generator(lz)
@static_generator(lz)