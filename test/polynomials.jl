@testset "polynomials" begin
    @test DiscretisedFluidQueues.gauss_lobatto_points(2.0,2.5,1)≈[2.25]
    @test DiscretisedFluidQueues.gauss_lobatto_points(2.0,3.0,2)≈[2.0;3.0]
    @test DiscretisedFluidQueues.gauss_lobatto_points(2.0,4.0,3)≈[2.0;3.0;4.0]
    nnodes = 5
    nodes = DiscretisedFluidQueues.gauss_lobatto_points(-1.0,1.0,nnodes)
    for n in 1:nnodes
        en = zeros(nnodes)
        en[n] = 1.0
        @test DiscretisedFluidQueues.lagrange_polynomials(nodes,nodes[n])≈en atol=sqrt(eps())
    end
    @test DiscretisedFluidQueues.lagrange_polynomials([1.0],1.0)≈[1.0] atol=sqrt(eps())
    @test DiscretisedFluidQueues.gauss_lobatto_weights(-1.0,1.0,1)≈[2.0]
    nnodes = 3
    nodes = DiscretisedFluidQueues.gauss_lobatto_points(-1.0,1.0,nnodes)
    @test DiscretisedFluidQueues.gauss_lobatto_weights(-1.0,1.0,nnodes)≈[1.0,4.0,1.0]./3
    x4 = x->x^4
    x4_interp = DiscretisedFluidQueues.lagrange_interpolation(x4,-1.0,1.0,5)
    x = collect(-3:0.02:3)
    @test x4_interp.(x)≈x4.(x)
    x4_quad = DiscretisedFluidQueues.gauss_lobatto_quadrature(x4,-1.0,1.0,4)
    @test x4_quad≈2/5

    x4_interp_p0 = DiscretisedFluidQueues.lagrange_interpolation(x->1.0-x4(x),-1.0,1.0,1)
    x = collect(-3:0.02:3)
    @test x4_interp_p0.(x)≈ones(size(x))
    x4_quad_p0 = DiscretisedFluidQueues.gauss_lobatto_quadrature(x->1.0-x4(x),-1.0,1.0,1)
    @test x4_quad_p0≈2.0
end 