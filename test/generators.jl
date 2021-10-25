P_lwr_reflecting = zeros(size(am.P_lwr))
P_lwr_reflecting[:,rates(am).>0.0] .= 1.0
P_lwr_reflecting = P_lwr_reflecting./sum(P_lwr_reflecting,dims=2)
P_upr_reflecting = zeros(size(am.P_upr))
P_upr_reflecting[:,rates(am).<0.0] .= 1.0
P_upr_reflecting = P_upr_reflecting./sum(P_upr_reflecting,dims=2)
refl_model = BoundedFluidQueue(am.T,am.S,P_lwr_reflecting,P_upr_reflecting)

rng = StableRNGs.StableRNG(94)
P_lwr_rand = rand(rng,size(am.P_lwr)...)
P_lwr_rand[:,(rates(am).==0.0).&DiscretisedFluidQueues._has_right_boundary(am.S)] .= 0.0
P_lwr_rand = P_lwr_rand./sum(P_lwr_rand,dims=2)
P_upr_rand = rand(rng,size(am.P_upr)...)
P_upr_rand[:,(rates(am).==0.0).&DiscretisedFluidQueues._has_left_boundary(am.S)] .= 0.0
P_upr_rand = P_upr_rand./sum(P_upr_rand,dims=2)
rand_model = BoundedFluidQueue(am.T,am.S,P_lwr_rand,P_upr_rand)

@testset "Generators" begin 
    @testset "augment_model" begin
        @testset "Lazy" begin 
            for i in (:dgmesh,:frapmesh), tmp_model in (:am,:refl_model,:rand_model)
                msh = @eval $i
                (i==:dgmesh) && include("test_data/"*string(typeof(msh.nodes))[1:4]*"/"*string(tmp_model)*"/test_DG_B_data.jl")
                (i==:frapmesh) && include("test_data/"*string(typeof(msh.nodes))[1:4]*"/"*string(tmp_model)*"/test_FRAP_B_data.jl")
                println((tmp_model,i))
                dq = @eval DiscretisedFluidQueue($tmp_model,$i)
                B = build_lazy_generator(dq)
                Matrix(B)
                # types
                @test typeof(B)<:LazyGenerator
                idx = DiscretisedFluidQueues.qbd_idx(dq)
                @test all(isapprox.(B,B_data[idx,idx],atol=1e-4))
                # multiplcation (values)
                @test all(DiscretisedFluidQueues.fast_mul(B,Matrix{Float64}(I(size(B,1)))) .≈ B)
                @test all(DiscretisedFluidQueues.fast_mul(Matrix{Float64}(I(size(B,1))),B) .≈ B)
                # row sums
                @test sum(abs.(sum(B,dims=2)))≈0.0 atol=√eps()
                @test all(isapprox.(B*B,B_data[idx,idx]*B_data[idx,idx],atol=1e-3))
                # multiplication (types)
                @test typeof(DiscretisedFluidQueues.fast_mul(B,Matrix{Float64}(I(size(B,1)))))==Array{Float64,2}
                @test typeof(DiscretisedFluidQueues.fast_mul(Matrix{Float64}(I(size(B,1))),B))==Array{Float64,2}
                @test typeof(DiscretisedFluidQueues.fast_mul(B,SparseMatrixCSC{Float64,Int}(I(size(B,1)))))==SparseMatrixCSC{Float64,Int}
                @test typeof(DiscretisedFluidQueues.fast_mul(SparseMatrixCSC{Float64,Int}(I(size(B,1))),B))==SparseMatrixCSC{Float64,Int}
                # size
                @test size(B) == (40,40)
                @test size(B,1) == 40
                @test size(B,2) == 40
                # getindex
                @testset "getindex" begin
                    sz = size(B,1)
                    ind = true
                    for i in 1:sz, j in 1:sz
                        ei = zeros(1,sz)
                        ei[i] = 1
                        ej = zeros(sz)[:,:]
                        ej[j] = 1
                        !(B[i,j] ≈ (ei*B*ej)[1]) && (ind = false)
                    end
                    @test ind
                end
            end
        end
        
        @testset "Full" begin
            for i in (:dgmesh,:frapmesh,:fvmesh), tmp_model in (:am,:refl_model,:rand_model)
                println((tmp_model,i))
                msh = @eval $i
                (i==:dgmesh) && include("test_data/"*string(typeof(msh.nodes))[1:4]*"/"*string(tmp_model)*"/test_DG_B_data.jl")
                (i==:frapmesh) && include("test_data/"*string(typeof(msh.nodes))[1:4]*"/"*string(tmp_model)*"/test_FRAP_B_data.jl")
                ((i==:fvmesh)&&(tmp_model==:am)) && include("test_data/"*string(typeof(msh.nodes))[1:4]*"/"*string(tmp_model)*"/test_FV_B_data.jl")
                if (!(i==:fvmesh)||(tmp_model==:am))
                    @eval begin 
                        dq = DiscretisedFluidQueue($tmp_model,$i)
                        B_Full = build_full_generator(dq)
                        if !(typeof($i)<:FVMesh)
                            B = build_lazy_generator(dq) 
                            @test build_full_generator(B)==B_Full
                            @test all(isapprox.(B_Full*B_Full,B*B,atol=√eps()))
                            # size
                            @test size(B_Full) == (40,40)
                            @test size(B_Full,1) == 40
                            @test size(B_Full,2) == 40
                            #row sums
                            @test all(isapprox.(sum(B_Full,dims=2),0,atol=√eps()))
                        else
                            @test size(B_Full) == (16,16)
                            @test size(B_Full,1) == 16
                            @test size(B_Full,2) == 16
                        end
                        @test typeof(B_Full.B)==SparseMatrixCSC{Float64,Int}
                        # types
                        B_data = B_data[DiscretisedFluidQueues.qbd_idx(dq),DiscretisedFluidQueues.qbd_idx(dq)]
                        @test all(isapprox.(B_Full,B_data,atol=1e-4))
                        # multiplcation (values)
                        @test B_Full*SparseMatrixCSC{Float64,Int}(I(size(B_Full,1)))==B_Full
                        @test B_Full==SparseMatrixCSC{Float64,Int}(I(size(B_Full,1)))*B_Full
                        # row sums
                        @test all(isapprox.(B_Full*B_Full,B_data*B_data,atol=1e-3))
                    end
                end
            end
        end
    end
    @testset "normal model" begin
        @testset "Lazy" begin 
            for i in (:dgmesh,:frapmesh) 
                println(i)
                msh = @eval $i
                dq = @eval DiscretisedFluidQueue(model,$i)
                B = build_lazy_generator(dq)
                # types
                @test typeof(B)<:LazyGenerator
                # multiplcation (values)
                @test all(DiscretisedFluidQueues.fast_mul(B,Matrix{Float64}(I(size(B,1)))) .≈ B)
                @test all(DiscretisedFluidQueues.fast_mul(Matrix{Float64}(I(size(B,1))),B) .≈ B)
                # row sums
                @test all(isapprox.(sum(B,dims=2),0,atol=√eps()))
                # multiplication (types)
                @test typeof(DiscretisedFluidQueues.fast_mul(B,Matrix{Float64}(I(size(B,1)))))==Array{Float64,2}
                @test typeof(DiscretisedFluidQueues.fast_mul(Matrix{Float64}(I(size(B,1))),B))==Array{Float64,2}
                @test typeof(DiscretisedFluidQueues.fast_mul(B,SparseMatrixCSC{Float64,Int}(I(size(B,1)))))==SparseMatrixCSC{Float64,Int}
                @test typeof(DiscretisedFluidQueues.fast_mul(SparseMatrixCSC{Float64,Int}(I(size(B,1))),B))==SparseMatrixCSC{Float64,Int}
                # size
                @test size(B) == (31,31)
                @test size(B,1) == 31
                @test size(B,2) == 31
                # getindex
                @testset "getindex" begin
                    sz = size(B,1)
                    getindex_does_not_match_mul = true
                    for i in 1:sz, j in 1:sz
                        ei = zeros(1,sz)
                        ei[i] = 1
                        ej = zeros(sz)[:,:]
                        ej[j] = 1
                        !(B[i,j] .≈ (ei*B*ej)[1]) && (getindex_does_not_match_mul = false)
                    end
                    @test getindex_does_not_match_mul
                end
            end
        end
        
        @testset "Full" begin
            for i in (:dgmesh,:frapmesh,:fvmesh)
                println(i)
                msh = @eval $i
                @eval begin 
                    dq = DiscretisedFluidQueue(model,$i)
                    B_Full = build_full_generator(dq)
                    if !(typeof($i)<:FVMesh)
                        B = build_lazy_generator(dq) 
                        @test build_full_generator(B)==B_Full
                        @test all(isapprox.(B_Full*B_Full,B*B,atol=√eps()))
                        # size
                        @test size(B_Full) == (31,31)
                        @test size(B_Full,1) == 31
                        @test size(B_Full,2) == 31
                        #row sums
                        @test all(isapprox.(sum(B_Full,dims=2),0,atol=√eps()))
                    else 
                        @test size(B_Full) == (13,13)
                        @test size(B_Full,1) == 13
                        @test size(B_Full,2) == 13
                    end
                    @test typeof(B_Full.B)==SparseMatrixCSC{Float64,Int}
                    # types
                    # multiplcation (values)
                    @test B_Full*SparseMatrixCSC{Float64,Int}(I(size(B_Full,1)))==B_Full
                    @test B_Full==SparseMatrixCSC{Float64,Int}(I(size(B_Full,1)))*B_Full
                end
            end
        end
    end
end
@testset "bnded qs default" begin 
    P_lwr = zeros(sum(rates(model.S).<0.0),n_phases(model))
    P_lwr[end] = 1.0
    P_upr = zeros(sum(rates(model.S).>0.0),n_phases(model))
    P_upr[2] = 1.0
    model_bnd = BoundedFluidQueue(T,S,P_lwr,P_upr)

    dq = DiscretisedFluidQueue(model,dgmesh)
    dq_bnd = DiscretisedFluidQueue(model_bnd,dgmesh)
    B = build_lazy_generator(dq)
    B_bnd = build_lazy_generator(dq_bnd)
    @test B==B_bnd
    @test B_bnd*B_bnd≈B*B
    @test Matrix{Float64}(I(size(B_bnd,1)))*B_bnd≈Matrix{Float64}(I(size(B_bnd,1)))*B 
end 
