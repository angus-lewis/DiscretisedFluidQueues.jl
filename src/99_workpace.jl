include("StochasticFluidQueues/src/StochasticFluidQueues.jl")
using .StochasticFluidQueues 
using Test
T = [-2.5 2 0.5; 1 -2 1; 1 2 -3]
T_nz = T - [0.01 0 0;0 0 0;0 0 0]
T_warn = T - [0.000001 0 0;0 0 0;0 0 0]
C = [0.0; 2.0; -3.0]
m = [1;1;-1]
S = PhaseSet(C,m)
bounds = [0.0,12]


@testset "PhaseSet struct" begin
    @test S[1]==Phase(0.0,1)
    @test_throws DomainError PhaseSet([-1.0],[1]) 
    @test_throws DomainError PhaseSet([-1.0],[2]) 
    @test S == [Phase(C[i],m[i]) for i in 1:length(C)]
    @test n_phases(S)==3
    @test all([get_rates(S,i) for i in 1:length(C)].==C)
    @test get_rates(S)==C
    @test all([get_membership(S,i) for i in 1:length(m)].==m)
    @test get_membership(S)==m
    @test phases(S)==1:length(C)
    @test N₊(S)==sum(C.>=0)
    @test N₋(S)==sum(C.<=0)
    @test checksquare(T)===nothing
    @test_throws DomainError checksquare(T[1:2,:])
end

@testset "FluidQueue struct" begin
    model = FluidQueue(T,S,bounds)
    @test model.T==T
    @test n_phases(model)==length(C)
    @test n_phases(model.S)==length(C)
    @test model.S==S
    @test get_rates(model)==C
    @test get_rates(model,1)==C[1]
    @test n_phases(model)==length(C)
    @test phases(model)==1:length(C)

    @test_logs (:warn,"row sums of T must be 0 (tol=1e-5)") FluidQueue(T_warn,S,bounds)

    @test_throws DomainError FluidQueue(T_nz[1:2,:],S,bounds)
    @test_throws DomainError FluidQueue(T_nz,S,bounds)
    @test_throws DomainError FluidQueue(T,S,[0])
    @test_throws DomainError FluidQueue(T,S[1:end-1],bounds)
end


model = FluidQueue(T,S,bounds) 
T_aug = [-2.5 0 2 0.5; 0 -2.5 2 0.5; 1 0 -2 1; 0 1 2 -3]
C_aug = [0;C]
m_aug = [1;-1;1;-1]
S_aug = PhaseSet(C_aug,m_aug)
am = FluidQueue(T_aug,S_aug,model.bounds)
for f in fieldnames(FluidQueue)
    @eval @test am.$f==augment_model(model).$f
end

empty_dgmesh = DGMesh()
nodes = [3;4;9;12]
nbases = 2
dgmesh = DGMesh(nodes,nbases)
@test typeof(dgmesh)==DGMesh
@test typeof(dgmesh)<:Mesh
@test n_intervals(dgmesh)==length(nodes)-1
@test Δ(dgmesh) == nodes[2:end]-nodes[1:end-1]
@test Δ(dgmesh,1) == nodes[2]-nodes[1]
@test total_n_bases(dgmesh) == (length(nodes)-1)*nbases
# test MakeQBDidx

# test lazy_generators 
