@testset "distributions" begin 
    for mtype in (:dgmesh,:frapmesh,:fvmesh)
        for closing_vec in ( :normalised_closing_operator,
                :naive_normalised_closing_operator)#, :unnormalised_closing_operator) 
                
            # pdfs
            f(x,i) = (i-1)/12.0./sum(1:3)
            msh = @eval $mtype
            dq = DiscretisedFluidQueue(am,msh)
            d = SFMDistribution(f,dq)#point_mass(11.990,2,am,$i)#
            if mtype!=:fvmesh
                @test sum(d)≈1.0
            end
            f_rec = (mtype!==:frapmesh) ? pdf(d) : pdf(d,@eval($(Symbol(closing_vec,"_pdf"))))
            x = (0.01:0.4:11.99)'
            if mtype!=:frapmesh
                @test all(isapprox.( f_rec.(x,1:4)-f.(x,1:4), 0, atol=sqrt(eps()) ))
            end

            # cdfs
            cdf_rec = (mtype!==:frapmesh) ? cdf(d) : cdf(d,@eval($(Symbol(closing_vec,"_cdf"))))
            f_cdf(x,i) = f(x,i)*x
            @test sum(cdf_rec.(msh.nodes[end],1:4))≈1.0
            @test sum(cdf_rec.(msh.nodes[end]+1.0,1:4))≈sum(f_cdf.(msh.nodes[end],1:4))
            x = (-1:0.4:12.99)'
            @test all(isapprox.( f_cdf.(x,1:4), cdf_rec.(x,1:4), atol=5e-2 ))

            # point masses at boundaries
            d = d * 0.5
            d[1] = 0.25
            d[end] = 0.25
            f_cdf_2(x,i) = 0.25*(i∈(2))*(x>=msh.nodes[1])+0.5*f_cdf(x,i)*(x>=msh.nodes[1])+0.25*(x>=msh.nodes[end])*(i∈(3))
            cdf_rec_2 = (mtype!==:frapmesh) ? cdf(d) : cdf(d,@eval($(Symbol(closing_vec,"_cdf"))))
            @test sum(cdf_rec_2.(msh.nodes[end],1:4))≈1.0
            @test sum(cdf_rec_2.(msh.nodes[end]+1.0,1:4))≈sum(f_cdf.(msh.nodes[end],1:4))
            @test all(isapprox.( f_cdf_2.(x,1:4), cdf_rec_2.(x,1:4), atol=5e-2 ))

            @test_throws DomainError left_point_mass(1,dq)
            tst = zeros(n_phases(am)*n_bases_per_phase(msh)+N₋(am.S)+N₊(am.S))
            tst[1] = 1.0
            @test left_point_mass(2,dq)==tst
            
            @test_throws DomainError right_point_mass(2,dq)
            tst2 = zeros(n_phases(am)*n_bases_per_phase(msh)+N₋(am.S)+N₊(am.S))
            tst2[end] = 1.0
            @test right_point_mass(3,dq)==tst2
        end
        # test point masses not at boundaries...
    end
end 