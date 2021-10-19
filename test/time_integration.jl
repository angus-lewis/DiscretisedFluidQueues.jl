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

        nodes = collect(-1.0:2.0/49.0:1.0)

        mesh = DGMesh(nodes,2)

        dq = DiscretisedFluidQueue(one_D_model,mesh)
        B = build_full_generator(dq)
        B[end-1,1:n_bases_per_cell(mesh)] = 
            B[n_bases_per_cell(mesh),n_bases_per_cell(mesh)+1:2*n_bases_per_cell(mesh)] 
        B[end-1,end] = 0.0
        t = 10.0
        f(x,i) = sin(π*x)
        d0 = SFMDistribution(f,dq)
        dt = integrate_time(d0,B,t,StableRK4(0.001); limiter=GeneralisedMUSCL)
        dt_no_limit = integrate_time(d0,B,t,StableRK4(0.001); limiter=NoLimiter)

        # plot(x->pdf(d0)(x,1),nodes[1],nodes[end])
        # plot!(x->pdf(limit(d0))(x,1),nodes[1],nodes[end])
        plot(x->pdf(dt)(x,1),nodes[1],nodes[end])
        plot!(x->pdf(dt_no_limit)(x,1),nodes[1],nodes[end])
        plot!(x->sin(π*(x+t)),nodes[1],nodes[end])

    end
end
