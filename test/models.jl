@testset "Phase and BoundedFluidQueue" begin 
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

    @testset "BoundedFluidQueue struct" begin
        @test model.T==T
        @test n_phases(model)==length(C)
        @test n_phases(model.S)==length(C)
        @test model.S==S
        @test rates(model)==C
        @test rates(model,1)==C[1]
        @test n_phases(model)==length(C)
        @test phases(model)==1:length(C)

        @test_logs (:warn,"row sums of T should be 0") BoundedFluidQueue(T_warn,S,nodes[end])

        @test_throws DomainError BoundedFluidQueue(T_nz[1:2,:],S,nodes[end])
        @test_throws DomainError BoundedFluidQueue(T,S[1:end-1],nodes[end])
    end

    for f in fieldnames(BoundedFluidQueue)
        @eval @test am.$f==augment_model(model).$f
    end
end 