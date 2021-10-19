# using LinearAlgebra, SparseArrays #Revise
# include("src/DiscretisedFluidQueues.jl")
# import .DiscretisedFluidQueues 
# using Test

import Pkg
display(pwd())
# Pkg.activate("DiscretisedFluidQueues.jl/")
Pkg.develop(url=pwd()*"/DiscretisedFluidQueues.jl")
using DiscretisedFluidQueues

T = [-2.5 2 0.5; 1 -2 1; 1 2 -3]

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
nbases = 11
dgmesh = DiscretisedFluidQueues.DGMesh(nodes,nbases)

am = DiscretisedFluidQueues.augment_model(model)

# fv_order = 15
# fvmesh = DiscretisedFluidQueues.FVMesh(nodes,fv_order)

# order = 3
# frapmesh = DiscretisedFluidQueues.FRAPMesh(nodes,order)

dq = DiscretisedFluidQueues.DiscretisedFluidQueue(model,dgmesh)

lz = DiscretisedFluidQueues.build_lazy_generator(dq)

# @macroexpand DiscretisedFluidQueues.@static_generator(lz)
# @static_generator(lz)

d0 = interior_point_mass(eps(),1,dq)
display(d0)

import Base: *, getindex, setindex!, size
using SparseArrays

struct MT <: AbstractMatrix{Float64}; a::SparseMatrixCSC{Float64,Int}; end

size(m::MT) = size(m.a)
getindex(m::MT,i) = m.a[i]
getindex(m::MT,i,j) = m.a[i,j]
setindex!(m::MT,x,i) = (m.a[i]=x)
setindex!(m::MT,x,i,j) = (m.a[i,j]=x)

using LinearAlgebra

a = diagm(0=>1:100.0, 1=>100:-1:2, -1=>11:109)
mt = MT(SparseMatrixCSC(a))

using BenchmarkTools

@btime a*a

for t in (:(Matrix{Float64}), :(SparseMatrixCSC{Float64,Int}))
    @eval fast_mul(a::MT,b::$t) = a.a*b
end

@btime fast_mul(mt,mt.a)

using DiscretisedFluidQueues, LinearAlgebra, Plots
order = 21
cme = ConcentratedMatrixExponential(order)
M = sylvester(Matrix(cme.S),Matrix(cme.S),cme.s*cme.a) 
me_rec0 = MatrixExponential(cme.a/cme.D,Matrix(cme.S),(I(order)-exp(cme.S*2))^-1*cme.s,cme.D)
f0(x) = cdf(me_rec0,x)
f0_rap(x) = 1-cdf(cme,1.0-x)*(x<1.0)

orbit_fun = DiscretisedFluidQueues.orbit(cme)
me_rec1 = MatrixExponential(orbit_fun(0.5)/cme.D,Matrix(cme.S),cme.s,cme.D)
me_rap1 = MatrixExponential(orbit_fun(0.5),Matrix(cme.S),(I(order)-exp(cme.S*2))^-1*cme.s,cme.D)
f1(x) = cdf(me_rec1,x)
f1_rap(x) = 1-cdf(me_rap1,1.0-x)*(x<1.0)


plot(f0,0,1.5)
plot!(f1,0,1.5)
plot!(f0_rap,0,1.5)
plot!(f1_rap,0,1.5,ylim=(-1,2))
