@testset "numerical checks" begin
    T = [-2.5 2 0.5; 1 -3 2; 1 2 -3]

    C = [0, 2.0, -6.0]
    S = PhaseSet(C)

    model = BoundedFluidQueue(T,S)
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
                generator[:,1] .= [
                    ones(N₋(model.S));
                    repeat(Δ(msh),n_phases(model));
                    ones(N₊(model.S))]
            else
                generator[:,1] .= 1.0
            end
            stationary_coeffs = b/generator.B
            d = SFMDistribution(stationary_coeffs[:],dq)
            stationary_cdf_estimate = (mtype!=FRAPMesh) ? cdf(d) : cdf(d, normalised_closing_operator_cdf)
            (mtype==FRAPMesh) && (stationary_cdf_estimate_naive = cdf(d, naive_normalised_closing_operator_cdf))
            analytical_cdf = stationary_distribution_x(model)[3]
            x_vec = nodes[1]:0.23:nodes[end]
            @testset "numerics - cdf" begin 
                for x in x_vec
                    @test isapprox(analytical_cdf(x),stationary_cdf_estimate.(x,1:3), rtol=1e-2)
                    (mtype==FRAPMesh) && (@test isapprox(analytical_cdf(x),stationary_cdf_estimate_naive.(x,1:3), rtol=1e-2))
                end
            end

            ####
            # augmented model
            ####
            dq_am = DiscretisedFluidQueue(am,msh)
            generator_am = build_full_generator(dq_am)
            b_am = zeros(1,size(generator_am,1))
            b_am[1] = 1.0
            if mtype==FVMesh
                generator_am[:,1] .= [
                    ones(N₋(am.S));
                    repeat(Δ(msh),n_phases(am));
                    ones(N₊(am.S))]
            else
                generator_am[:,1] .= 1.0
            end
            stationary_coeffs_am = b_am/generator_am.B
            d_am = SFMDistribution(stationary_coeffs_am[:],dq_am)
            stationary_cdf_estimate_am = (mtype!=FRAPMesh) ? cdf(d_am) : (stationary_cdf_estimate_am_naive = cdf(d_am, normalised_closing_operator_cdf))
            (mtype==FRAPMesh) && (stationary_cdf_estimate_am_naive = cdf(d_am, naive_normalised_closing_operator_cdf))
            analytical_cdf_am = stationary_distribution_x(am)[3]
            x_vec = nodes[1]:0.23:nodes[end]
            @testset "numerics - augmented model - cdf" begin
                for x in x_vec
                    @test isapprox(analytical_cdf_am(x),stationary_cdf_estimate_am.(x,1:4), rtol=1e-2)
                    (mtype==FRAPMesh) && (@test isapprox(analytical_cdf_am(x),stationary_cdf_estimate_am_naive.(x,1:4), rtol=1e-2))
                end
            end

            @testset "numerics - augmented model vs vanilla cdf" begin
                for x in x_vec
                    @test isapprox(stationary_cdf_estimate.(x,2:3),stationary_cdf_estimate_am.(x,3:4), rtol=1e-2 )
                    @test isapprox(stationary_cdf_estimate.(x,1),sum(stationary_cdf_estimate_am.(x,1:2)), rtol=1e-2 )
                    (mtype==FRAPMesh) && (@test isapprox(stationary_cdf_estimate_naive.(x,2:3),stationary_cdf_estimate_am_naive.(x,3:4), rtol=1e-2 ))
                    (mtype==FRAPMesh) && (@test isapprox(stationary_cdf_estimate_naive.(x,1),sum(stationary_cdf_estimate_am_naive.(x,1:2)), rtol=1e-2 ))
                end
            end
        end
    end
end