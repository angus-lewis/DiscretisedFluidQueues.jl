@testset "time integration" begin
    D = [-1.0 2.0*π; -2.0*π -1.0]
    x0 = [1.0 0.0]
    test_data = x0*exp(D)
    e = ForwardEuler(1e-3)
    heuns = Heuns(1e-3)
    ssprk3 = StableRK3(1e-3)
    ssprk4 = StableRK4(1e-3)
    @test test_data≈integrate_time(x0,D,1.0,e) atol=e.step_size*10.0
    @test test_data≈integrate_time(x0,D,1.0,heuns) atol=heuns.step_size*10.0
    @test test_data≈integrate_time(x0,D,1.0,ssprk3) atol=ssprk3.step_size*10.0
    @test test_data≈integrate_time(x0,D,1.0,ssprk3) atol=ssprk4.step_size*10.0

    @testset "limiter -- MUSCL" begin 
        using DiscretisedFluidQueues
        one_D_T = [0.0][:,:]
        one_D_c = [1.0]
        one_D_model = FluidQueue(one_D_T,one_D_c)

        nodes = collect(0.0:0.1:5.0);

        mesh = DGMesh(nodes,5)

        dq = DiscretisedFluidQueue(one_D_model,mesh)
        B = build_full_generator(dq)

        f(x,i) = 1(x>1.1)#sin(2*π*x)
        d0 = SFMDistribution(f,dq)
        dt = integrate_time(d0,B,1.0,StableRK4(0.01); limiter=GeneralisedMUSCL)
        dt_no_limit = integrate_time(d0,B,1.0,StableRK4(0.01); limiter=NoLimiter)
        # plot(x->pdf(d0)(x,1),0,5)
        # plot!(x->pdf(limit(d0))(x,1),0,5)
        # plot!(x->pdf(dt)(x,1),0,5)
        # plot!(x->pdf(dt_no_limit)(x,1),0,5)

    end
end
