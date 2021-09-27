@testset "numerical checks" begin
    T = [-2.5 2 0.5; 1 -3 2; 1 2 -3]

    C = [0, 2.0, -6.0]
    S = PhaseSet(C)

    model = FluidQueue(T,S)
    am = augment_model(model)

    mtypes = (DGMesh,
        FVMesh,
        FRAPMesh)
    for mtype in mtypes
        @testset "mtype" begin
            nodes = collect(0:0.1:12)
            order = 5
            msh = mtype(nodes,order)

            ####
            # non-augmented model
            ####
            dq = DiscretisedFluidQueue(model,msh)
            generator = build_full_generator(dq)
            b = zeros(1,size(generator,1))
            b[1] = 1.0
            if mtype==FVMesh
                generator[:,1] = [
                    ones(N₋(model.S));
                    repeat(Δ(msh),n_phases(model));
                    ones(N₊(model.S))]
            else
                generator[:,1] .= 1.0
            end
            stationary_coeffs = b/generator.B
            d = SFMDistribution(stationary_coeffs,dq)
            stationary_cdf_estimate = cdf(d)
            analytical_cdf = stationary_distribution_x(model)[3]
            x_vec = nodes[1]:0.23:nodes[end]
            pass = true
            for x in x_vec
                (!isapprox(analytical_cdf(x),stationary_cdf_estimate.(x,1:3), rtol=1e-2)) && (pass = false)
            end
            @test pass

            ####
            # augmented model
            ####
            dq_am = DiscretisedFluidQueue(am,msh)
            generator_am = build_full_generator(dq_am)
            b_am = zeros(1,size(generator_am,1))
            b_am[1] = 1.0
            if mtype==FVMesh
                generator_am[:,1] = [
                    ones(N₋(am.S));
                    repeat(Δ(msh),n_phases(am));
                    ones(N₊(am.S))]
            else
                generator_am[:,1] .= 1.0
            end
            stationary_coeffs_am = b_am/generator_am.B
            d_am = SFMDistribution(stationary_coeffs_am,dq_am)
            stationary_cdf_estimate_am = cdf(d_am)
            analytical_cdf_am = stationary_distribution_x(am)[3]
            x_vec = nodes[1]:0.23:nodes[end]
            pass = true
            for x in x_vec
                (!isapprox(analytical_cdf_am(x),stationary_cdf_estimate_am.(x,1:4), rtol=1e-2)) && (pass = false)
            end
            @test pass

            pass = true
            for x in x_vec
                (!isapprox(stationary_cdf_estimate.(x,2:3),stationary_cdf_estimate_am.(x,3:4), rtol=1e-2 )) && (pass=false)
                (!isapprox(stationary_cdf_estimate.(x,1),sum(stationary_cdf_estimate_am.(x,1:2)), rtol=1e-2 )) && (pass=false)
            end
            @test pass
        end
    end
end