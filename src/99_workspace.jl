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

P_lwr = zeros(sum(rates(model.S).<0.0),n_phases(model))
# P_lwr[end] = 1.0
P_lwr[:] .= 1/3
P_upr = zeros(sum(rates(model.S).>0.0),n_phases(model))
# P_upr[2] = 1.0
P_upr .= 1/3
model_bnd = BoundedFluidQueue(T,S,P_lwr,P_upr)

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
# (i==:dgmesh) && include("test/test_DG_B_data.jl")
# (i==:frapmesh) && include("test/test_FRAP_B_data.jl")
dq = @eval DiscretisedFluidQueue(model,$i)
dq_bnd = @eval DiscretisedFluidQueue(model_bnd,$i)
B = build_lazy_generator(dq)
B_bnd = build_lazy_generator(dq_bnd)

@test all(fast_mul(B,Matrix{Float64}(I(size(B,1)))) .== B)

# B = build_full_generator(dq)