@testset "Phase and FluidQueue" begin 
    @testset "PhaseSet struct" begin
        @test S[1]==Phase(0.0,1)
        @test_throws DomainError PhaseSet([-1.0],[1]) 
        @test_throws DomainError PhaseSet([-1.0],[2]) 
        @test S == [Phase(C[i]) for i in 1:length(C)]
        @test n_phases(S)==3
        @test [rates(S,i) for i in 1:length(C)]==C
        @test rates(S)==C
        @test [membership(S,i) for i in 1:length(m)]==m
        @test membership(S)==m
        @test phases(S)==1:length(C)
        @test N₊(S)==sum(C.>=0.0)
        @test N₋(S)==sum(C.<=0.0)
        @test DiscretisedFluidQueues.checksquare(T)===nothing
        @test_throws DomainError DiscretisedFluidQueues.checksquare(T[1:2,:])
    end

    @testset "FluidQueue struct" begin
        @test model.T==T
        @test n_phases(model)==length(C)
        @test n_phases(model.S)==length(C)
        @test model.S==S
        @test rates(model)==C
        @test rates(model,1)==C[1]
        @test n_phases(model)==length(C)
        @test phases(model)==1:length(C)

        @test_logs (:warn,"row sums of T must be 0 (tol=1e-5)") FluidQueue(T_warn,S)

        @test_throws DomainError FluidQueue(T_nz[1:2,:],S)
        @test_throws DomainError FluidQueue(T_nz,S)
        @test_throws DomainError FluidQueue(T,S[1:end-1])
    end

    for f in fieldnames(FluidQueue)
        @eval @test am.$f==augment_model(model).$f
    end
end 