@testset "Generators" begin 
    @testset "Lazy" begin 
        for i in (:dgmesh,:frapmesh) 
            (i==:dgmesh) && include("test_DG_B_data.jl")
            (i==:frapmesh) && include("test_FRAP_B_data.jl")
            dq = @eval DiscretisedFluidQueue(am,$i)
            B = build_lazy_generator(dq)
            # types
            @test typeof(B)==LazyGenerator
            @test all(isapprox.(B,B_data,atol=1e-4))
            # multiplcation (values)
            @test all(fast_mul(B,Matrix{Float64}(I(size(B,1)))) .== B)
            @test all(fast_mul(Matrix{Float64}(I(size(B,1))),B) .== B)
            @test all(fast_mul(Matrix{Float64}(I(size(B,1))),B) .== B)
            # row sums
            @test all(isapprox.(sum(B,dims=2),0,atol=√eps()))
            @test all(isapprox.(B*B,B_data*B_data,atol=1e-3))
            # multiplication (types)
            @test typeof(fast_mul(B,Matrix{Float64}(I(size(B,1)))))==Array{Float64,2}
            @test typeof(fast_mul(Matrix{Float64}(I(size(B,1))),B))==Array{Float64,2}
            @test typeof(fast_mul(B,SparseMatrixCSC{Float64,Int}(I(size(B,1)))))==SparseMatrixCSC{Float64,Int}
            @test typeof(fast_mul(SparseMatrixCSC{Float64,Int}(I(size(B,1))),B))==SparseMatrixCSC{Float64,Int}
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
                    ej = zeros(sz)
                    ej[j] = 1
                    !(B[i,j] == (ei*B*ej)[1]) && (ind = false)
                end
                @test ind
            end
        end
    end
    
    @testset "Full" begin
        for i in (:dgmesh,:frapmesh,:fvmesh)
            (i==:dgmesh) && include("test_DG_B_data.jl")
            (i==:frapmesh) && include("test_FRAP_B_data.jl")
            (i==:fvmesh) && include("test_FV_B_data.jl")
            @eval begin 
                dq = DiscretisedFluidQueue(am,$i)
                B_Full = build_full_generator(dq)
                if typeof($i)!=FVMesh
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