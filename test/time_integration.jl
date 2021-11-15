@testset "time integration" begin
    D = [-1.0 2.0*π; -2.0*π -1.0]
    x0 = [1.0 0.0]
    test_data = (x0*exp(D))[:]
    e = ForwardEuler(1e-3)
    heuns = Heuns(1e-3)
    ssprk3 = StableRK3(1e-3)
    ssprk4 = StableRK4(1e-3)
    @test test_data≈integrate_time(x0,D,1.0,e) atol=e.step_size*10.0
    @test test_data≈integrate_time(x0,D,1.0,heuns) atol=heuns.step_size*10.0
    @test test_data≈integrate_time(x0,D,1.0,ssprk3) atol=ssprk3.step_size*10.0
    @test test_data≈integrate_time(x0,D,1.0,ssprk3) atol=ssprk4.step_size*10.0

    @testset "limiter -- MUSCL" begin 
        # using DiscretisedFluidQueues
        one_D_T = [0.0][:,:]
        one_D_c = [1.0]
        one_D_model = BoundedFluidQueue(one_D_T,one_D_c,1.0)

        nodes = collect(-1.0:2.0/50.0:1.0)

        mesh = DGMesh(nodes,7)#FRAPMesh(nodes,7)#

        dq = DiscretisedFluidQueue(one_D_model,mesh)
        B = build_full_generator(dq)
        # B[2,(end-n_bases_per_cell(mesh)+1):end] = 
        #     B[n_bases_per_cell(mesh)+2,2:n_bases_per_cell(mesh)+1] 
        # B[2,1] = 0.0
        t = 1.0
        f(x,i) = Float64(x<-0.5)#sin(π*x)#
        d0 = SFMDistribution(f,dq)# interior_point_mass(-0.5,1,dq)#

        pdf0 = pdf(d0)
        pdf0_limit = pdf(limit(d0))
        x_vals = -0.55:0.001:-0.45
        @test any(pdf0.(x_vals,1).>1.01)
        @test !any(pdf0_limit.(x_vals,1).>1.01)
        @test any(pdf0.(x_vals,1).<-0.01)
        @test !any(pdf0_limit.(x_vals,1).<-0.01)

        dt = integrate_time(d0,B,t,StableRK4(0.01))
        dt_no_limit = integrate_time(d0,B,t,StableRK4(0.01); limiter=NoLimiter)

        pdft = pdf(dt)
        pdft_no_limit = pdf(limit(dt_no_limit))
        x_vals = -0.05:0.001:0.55
        @test !any(pdft.(x_vals,1).>1.01)
        @test any(pdft_no_limit.(x_vals,1).>1.01)
        @test !any(pdft.(x_vals,1).<-0.01)
        @test any(pdft_no_limit.(x_vals,1).<-0.01)
    end
end
