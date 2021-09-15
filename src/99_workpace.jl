using Revise, LinearAlgebra, SparseArrays
include("StochasticFluidQueues.jl")
import .StochasticFluidQueues 
using Test

T = [-2.5 2 0.5; 1 -2 1; 1 2 -3]
T_nz = T - [0.01 0 0;0 0 0;0 0 0]
T_warn = T - [0.000001 0 0;0 0 0;0 0 0]

C = [0.0; 2.0; -3.0]
m = [1;1;-1]
S = StochasticFluidQueues.PhaseSet(C,m)

bounds = [0.0,12]
model = StochasticFluidQueues.FluidQueue(T,S,bounds)

T_aug = [-2.5 0 2 0.5; 0 -2.5 2 0.5; 1 0 -2 1; 0 1 2 -3]
C_aug = [0;C]
m_aug = [1;-1;1;-1]
S_aug = StochasticFluidQueues.PhaseSet(C_aug,m_aug)
am = StochasticFluidQueues.FluidQueue(T_aug,S_aug,model.bounds)

nodes = [0.0;3.0;4.0;12.0]
nbases = 3
dgmesh = StochasticFluidQueues.DGMesh(nodes,nbases)

am = StochasticFluidQueues.augment_model(model)

fv_order = 3
fvmesh = StochasticFluidQueues.FVMesh(nodes,fv_order)

order = 3
frapmesh = StochasticFluidQueues.FRAPMesh(nodes,order)

@testset begin
@testset "Phase and FluidQueue" begin 
    @testset "PhaseSet struct" begin
        @test S[1]==StochasticFluidQueues.Phase(0.0,1)
        @test_throws DomainError StochasticFluidQueues.PhaseSet([-1.0],[1]) 
        @test_throws DomainError StochasticFluidQueues.PhaseSet([-1.0],[2]) 
        @test S == [StochasticFluidQueues.Phase(C[i],m[i]) for i in 1:length(C)]
        @test StochasticFluidQueues.n_phases(S)==3
        @test all([StochasticFluidQueues.rates(S,i) for i in 1:length(C)].==C)
        @test StochasticFluidQueues.rates(S)==C
        @test all([StochasticFluidQueues.membership(S,i) for i in 1:length(m)].==m)
        @test StochasticFluidQueues.membership(S)==m
        @test StochasticFluidQueues.phases(S)==1:length(C)
        @test StochasticFluidQueues.N₊(S)==sum(m.>=0)
        @test StochasticFluidQueues.N₋(S)==sum(m.<=0)
        @test StochasticFluidQueues.checksquare(T)===nothing
        @test_throws DomainError StochasticFluidQueues.checksquare(T[1:2,:])
    end

    @testset "FluidQueue struct" begin
        @test model.T==T
        @test StochasticFluidQueues.n_phases(model)==length(C)
        @test StochasticFluidQueues.n_phases(model.S)==length(C)
        @test model.S==S
        @test StochasticFluidQueues.rates(model)==C
        @test StochasticFluidQueues.rates(model,1)==C[1]
        @test StochasticFluidQueues.n_phases(model)==length(C)
        @test StochasticFluidQueues.phases(model)==1:length(C)

        @test_logs (:warn,"row sums of T must be 0 (tol=1e-5)") StochasticFluidQueues.FluidQueue(T_warn,S,bounds)

        @test_throws DomainError StochasticFluidQueues.FluidQueue(T_nz[1:2,:],S,bounds)
        @test_throws DomainError StochasticFluidQueues.FluidQueue(T_nz,S,bounds)
        @test_throws DomainError StochasticFluidQueues.FluidQueue(T,S,[0])
        @test_throws DomainError StochasticFluidQueues.FluidQueue(T,S[1:end-1],bounds)
    end

    for f in fieldnames(StochasticFluidQueues.FluidQueue)
        @eval @test am.$f==StochasticFluidQueues.augment_model(model).$f
    end
end 

@testset "Mesh Basics" begin 
    @testset "DG Mesh basics" begin    
        @test typeof(dgmesh)==StochasticFluidQueues.DGMesh
        @test typeof(dgmesh)<:StochasticFluidQueues.Mesh
        @test StochasticFluidQueues.n_intervals(dgmesh)==length(nodes)-1
        @test StochasticFluidQueues.Δ(dgmesh) == nodes[2:end]-nodes[1:end-1]
        @test StochasticFluidQueues.Δ(dgmesh,1) == nodes[2]-nodes[1]
        @test StochasticFluidQueues.total_n_bases(dgmesh) == (length(nodes)-1)*nbases
        @test StochasticFluidQueues.n_bases(dgmesh) == 3
        @test StochasticFluidQueues.cell_nodes(dgmesh)≈
            [nodes[1:end-1]';(nodes[1:end-1]'+nodes[2:end]')/2;nodes[2:end]'] atol=1e-5
        @test StochasticFluidQueues.basis(dgmesh) == "lagrange"
        # @test local_dg_operators(dgmesh) == ???
        # test MakeQBDidx 
    end

    @testset "FV Mesh basics" begin    
        @test typeof(fvmesh)==StochasticFluidQueues.FVMesh
        @test typeof(fvmesh)<:StochasticFluidQueues.Mesh
        @test StochasticFluidQueues.n_intervals(fvmesh)==length(nodes)-1
        @test StochasticFluidQueues.Δ(fvmesh) == nodes[2:end]-nodes[1:end-1]
        @test StochasticFluidQueues.Δ(fvmesh,1) == nodes[2]-nodes[1]
        @test StochasticFluidQueues.total_n_bases(fvmesh) == (length(nodes)-1)
        @test StochasticFluidQueues.n_bases(fvmesh) == 1
        @test StochasticFluidQueues._order(fvmesh) == fv_order
        @test StochasticFluidQueues.cell_nodes(fvmesh)≈Array(((fvmesh.nodes[1:end-1] + fvmesh.nodes[2:end]) / 2 )') atol=1e-5
        @test StochasticFluidQueues.basis(fvmesh) == ""
    end

    @testset "FRAP Mesh basics" begin    
        @test typeof(frapmesh)==StochasticFluidQueues.FRAPMesh
        @test typeof(frapmesh)<:StochasticFluidQueues.Mesh
        @test StochasticFluidQueues.n_intervals(frapmesh)==length(nodes)-1
        @test StochasticFluidQueues.Δ(frapmesh) == nodes[2:end]-nodes[1:end-1]
        @test StochasticFluidQueues.Δ(frapmesh,1) == nodes[2]-nodes[1]
        @test StochasticFluidQueues.total_n_bases(frapmesh) == (length(nodes)-1)*order
        @test StochasticFluidQueues.n_bases(frapmesh) == order
        @test StochasticFluidQueues.cell_nodes(frapmesh)≈Array(((frapmesh.nodes[1:end-1] + frapmesh.nodes[2:end]) / 2 )') atol=1e-5
        @test StochasticFluidQueues.basis(frapmesh) == ""
    end
end 

@testset "Generators" begin 
    @testset "Lazy" begin 
        for i in (:dgmesh,:frapmesh) 
            (i==:dgmesh) && include("test_DG_B_data.jl")
            (i==:frapmesh) && include("test_FRAP_B_data.jl")
            @eval begin
                B = StochasticFluidQueues.MakeLazyGenerator(am,$i)
                # types
                @test typeof(B)==StochasticFluidQueues.LazyGenerator
                @test typeof(B)<:AbstractArray
                @test all(isapprox.(B,B_data,atol=1e-4))
                # multiplcation (values)
                @test B*Matrix(I(size(B,1))) == B
                @test Matrix(I(size(B,1)))*B == B
                @test Matrix(I(size(B,1)))*B == B
                # row sums
                @test all(isapprox.(sum(B,dims=2),0,atol=√eps()))
                @test all(isapprox.(B*B,B_data*B_data,atol=1e-3))
                # multiplication (types)
                @test typeof(B*Matrix(I(size(B,1))))==Array{Float64,2}
                @test typeof(Matrix(I(size(B,1)))*B)==Array{Float64,2}
                @test typeof(B*SparseMatrixCSC{Float64,Int}(I(size(B,1))))==SparseMatrixCSC{Float64,Int}
                @test typeof(SparseMatrixCSC{Float64,Int}(I(size(B,1)))*B)==SparseMatrixCSC{Float64,Int}
                @test typeof(B*B)==SparseMatrixCSC{Float64,Int}
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
                # subtration
                @test all(isapprox.(B-B*Matrix(I(size(B,1))),0,atol=sqrt(eps())))
            end
        end
    end
    
    @testset "Full" begin
        for i in (:dgmesh,:frapmesh,:fvmesh)
            (i==:dgmesh) && include("test_DG_B_data.jl")
            (i==:frapmesh) && include("test_FRAP_B_data.jl")
            (i==:fvmesh) && include("test_FV_B_data.jl")
            @eval begin 
                B_Full = StochasticFluidQueues.MakeFullGenerator(am,$i)
                if typeof($i)!=StochasticFluidQueues.FVMesh
                    B = StochasticFluidQueues.MakeLazyGenerator(am,$i) 
                    @test StochasticFluidQueues.materialise(B)==B_Full
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

# test 5_SFM_operators

# test 6_ME_tools
@testset "ME tools" begin 
    @test StochasticFluidQueues.MatrixExponential <: StochasticFluidQueues.AbstractMatrixExponential
    @test_throws DimensionMismatch StochasticFluidQueues.MatrixExponential(ones(1,2),-ones(1,1),ones(1,1))
    @test_throws DomainError StochasticFluidQueues.MatrixExponential(ones(1,1),-[1.0 1.0],ones(1,1))
    @test_throws DomainError StochasticFluidQueues.MatrixExponential(ones(1,1),-[1.0 1.0],ones(1,2))
    exp_rv = StochasticFluidQueues.MatrixExponential(ones(1,1),-ones(1,1),ones(1,1))
    # pdf 
    @test StochasticFluidQueues._order(exp_rv)==1
    exp_pdf = StochasticFluidQueues.pdf(exp_rv)
    @test typeof(exp_pdf(1.0))<:Float64
    r = -log.(rand(10))
    @test exp_pdf.(r)≈exp.(-r)
    @test_throws DomainError StochasticFluidQueues.pdf([1.0 1.0],exp_rv)
    @test StochasticFluidQueues.pdf(exp_rv,r)≈StochasticFluidQueues.pdf(exp_rv).(r)
    @test StochasticFluidQueues.pdf(exp(-1).*ones(1,1),exp_rv,r)≈StochasticFluidQueues.pdf(exp_rv).(r.+1.0)
    # ccdf
    exp_ccdf = StochasticFluidQueues.ccdf(exp_rv)
    @test typeof(exp_ccdf(1.0))<:Float64
    r = -log.(rand(10))
    @test exp_ccdf.(r)≈exp.(-r)
    @test_throws DomainError StochasticFluidQueues.ccdf([1.0 1.0],exp_rv)
    @test StochasticFluidQueues.ccdf(exp_rv,r)≈StochasticFluidQueues.ccdf(exp_rv).(r)
    @test StochasticFluidQueues.ccdf(exp(-1).*ones(1,1),exp_rv,r)≈StochasticFluidQueues.ccdf(exp_rv).(r.+1.0)
    # cdf
    exp_cdf = StochasticFluidQueues.cdf(exp_rv)
    @test typeof(exp_cdf(1.0))<:Float64
    r = -log.(rand(10))
    @test exp_cdf.(r)≈1.0.-exp.(-r)
    @test_throws DomainError StochasticFluidQueues.cdf([1.0 1.0],exp_rv)
    @test StochasticFluidQueues.cdf(exp_rv,r)≈StochasticFluidQueues.cdf(exp_rv).(r)
    # MakeME
    me_1 = StochasticFluidQueues.MakeME(StochasticFluidQueues.CMEParams[1])
    @test StochasticFluidQueues.cdf(me_1,r)≈StochasticFluidQueues.cdf(exp_rv,r)
    #
    me_3 = StochasticFluidQueues.MakeME(StochasticFluidQueues.CMEParams[3])
    @test typeof(me_3)<:StochasticFluidQueues.AbstractMatrixExponential
    @test typeof(me_3)<:StochasticFluidQueues.ConcentratedMatrixExponential
    me_3_cdf = StochasticFluidQueues.cdf(me_3)
    @test typeof(me_3_cdf(1.0))<:Float64
    for f in fieldnames(StochasticFluidQueues.ConcentratedMatrixExponential)
        @eval begin 
            me_3 = StochasticFluidQueues.MakeME(StochasticFluidQueues.CMEParams[3])
            @test StochasticFluidQueues.ConcentratedMatrixExponential(3).$f==me_3.$f
        end
    end
    @test -sum(me_3.a/me_3.S)≈1.0
    # MakeErlang
    erl_1 = StochasticFluidQueues.MakeErlang(1)
    @test typeof(erl_1)<:StochasticFluidQueues.AbstractMatrixExponential
    @test typeof(erl_1)<:StochasticFluidQueues.MatrixExponential
    @test StochasticFluidQueues.ccdf(erl_1,r)≈StochasticFluidQueues.ccdf(exp_rv,r)
    erl_3 = StochasticFluidQueues.MakeErlang(3)
    @test -sum(erl_3.a/erl_3.S)≈1.0
    e1 = zeros(1,3)
    e1[1]=1.0
    @test erl_3.a==e1
    @test erl_3.S≈[-3.0 3.0 0.0; 0.0 -3.0 3.0; 0.0 0.0 -3.0]
    @test erl_3.s≈3*e1[end:-1:1]
    # orbits 
    exp_rv_orbit_fun = StochasticFluidQueues.orbit(exp_rv)
    @test exp_rv_orbit_fun(2.0)≈[1.0]

    me_3_orbit_fun = StochasticFluidQueues.orbit(me_3)
    me_3_mean_2 = StochasticFluidQueues.ConcentratedMatrixExponential(3; mean=2.0)
    me_3_mean_2_orbit_fun = StochasticFluidQueues.orbit(me_3_mean_2)
    @test me_3_mean_2_orbit_fun.(r)≈me_3_orbit_fun.(0.5*r)
    me_3_not_CME_Type = StochasticFluidQueues.MatrixExponential(me_3.a,me_3.S,me_3.s,me_3.D)
    me_3_not_CME_Type_orbit_fun = StochasticFluidQueues.orbit(me_3_not_CME_Type)
    @test me_3_orbit_fun.(r)≈me_3_not_CME_Type_orbit_fun.(r)
    me_3_mean_2_not_CME_Type = 
        StochasticFluidQueues.MatrixExponential(me_3_mean_2.a,me_3_mean_2.S,me_3_mean_2.s,me_3_mean_2.D)
    me_3_mean_2_not_CME_Type_orbit_fun = StochasticFluidQueues.orbit(me_3_mean_2_not_CME_Type)
    @test me_3_mean_2_not_CME_Type_orbit_fun.(r)≈me_3_not_CME_Type_orbit_fun.(0.5*r)
    # Expected orbits
    ω = abs(me_3.S[3,2])
    period = 2*pi/ω
    exp_orbit = 
        StochasticFluidQueues.expected_orbit_from_pdf(
            x->StochasticFluidQueues.ccdf(me_3,x)/-sum(me_3.a/me_3.S),
            me_3,0.0,period,10000)/(1-exp(me_3.S[1,1]*period))
    stationary = (me_3.a/me_3.S)/sum(me_3.a/me_3.S)
    @test exp_orbit≈stationary atol=1e-6
    exp_orbit_from_cdf = 
        StochasticFluidQueues.expected_orbit_from_cdf(
            x->StochasticFluidQueues.cdf(stationary,me_3,x)/-sum(me_3.a/me_3.S),
            me_3,0.0,period,10000)/(1-exp(me_3.S[1,1]*period))
    @test exp_orbit_from_cdf≈stationary atol=1e-6
end 

@testset "polynomials" begin
    @test StochasticFluidQueues.gauss_lobatto_points(2.0,2.5,1)≈[2.25]
    @test StochasticFluidQueues.gauss_lobatto_points(2.0,3.0,2)≈[2.0;3.0]
    @test StochasticFluidQueues.gauss_lobatto_points(2.0,4.0,3)≈[2.0;3.0;4.0]
    nnodes = 5
    nodes = StochasticFluidQueues.gauss_lobatto_points(-1.0,1.0,nnodes)
    for n in 1:nnodes
        en = zeros(nnodes)
        en[n] = 1.0
        @test StochasticFluidQueues.lagrange_polynomials(nodes,nodes[n])≈en atol=sqrt(eps())
    end
    @test StochasticFluidQueues.lagrange_polynomials([1.0],1.0)≈[1.0] atol=sqrt(eps())
    @test StochasticFluidQueues.gauss_lobatto_weights(-1.0,1.0,1)≈[2.0]
    nnodes = 3
    nodes = StochasticFluidQueues.gauss_lobatto_points(-1.0,1.0,nnodes)
    @test StochasticFluidQueues.gauss_lobatto_weights(-1.0,1.0,nnodes)≈[1.0,4.0,1.0]./3
    x4 = x->x^4
    x4_interp = StochasticFluidQueues.lagrange_interpolation(x4,-1.0,1.0,5)
    x = collect(-3:0.02:3)
    @test x4_interp.(x)≈x4.(x)
    x4_quad = StochasticFluidQueues.gauss_lobatto_quadrature(x4,-1.0,1.0,4)
    @test x4_quad≈2/5

    x4_interp_p0 = StochasticFluidQueues.lagrange_interpolation(x->1.0-x4(x),-1.0,1.0,1)
    x = collect(-3:0.02:3)
    @test x4_interp_p0.(x)≈ones(size(x))
    x4_quad_p0 = StochasticFluidQueues.gauss_lobatto_quadrature(x->1.0-x4(x),-1.0,1.0,1)
    @test x4_quad_p0≈2.0
end 

# check numerical solutions (stationary?)
# test 11_distributions 

@testset "distributions" begin 
    for mtype in (:dgmesh,:frapmesh,:fvmesh)
        # pdfs
        f(x,i) = (i-1)/12.0./sum(1:3)
        msh = @eval $mtype
        d = StochasticFluidQueues.SFMDistribution(f,am,msh)#StochasticFluidQueues.point_mass(11.990,2,am,$i)#
        if mtype!=:fvmesh
            @test sum(d)≈1.0
        end
        f_rec = StochasticFluidQueues.pdf(d,am)
        x = (0.01:0.4:11.99)'
        if mtype!=:frapmesh
            @test all(isapprox.( f_rec.(x,1:4)-f.(x,1:4), 0, atol=sqrt(eps()) ))
        end

        # cdfs
        cdf_rec = StochasticFluidQueues.cdf(d,am)
        f_cdf(x,i) = f(x,i)*x
        @test sum(cdf_rec.(msh.nodes[end],1:4))≈1.0
        @test sum(cdf_rec.(msh.nodes[end]+1.0,1:4))≈sum(f_cdf.(msh.nodes[end],1:4))
        x = (-1:0.4:12.99)'
        @test all(isapprox.( f_cdf.(x,1:4), cdf_rec.(x,1:4), atol=5e-2 ))

        # point masses at boundaries
        d = StochasticFluidQueues.:*(d,0.5)
        d[1] = 0.25
        d[end] = 0.25
        f_cdf_2(x,i) = 0.25*(i∈(2))*(x>=msh.nodes[1])+0.5*f_cdf(x,i)*(x>=msh.nodes[1])+0.25*(x>=msh.nodes[end])*(i∈(3))
        cdf_rec_2 = StochasticFluidQueues.cdf(d,am)
        @test sum(cdf_rec_2.(msh.nodes[end],1:4))≈1.0
        @test sum(cdf_rec_2.(msh.nodes[end]+1.0,1:4))≈sum(f_cdf.(msh.nodes[end],1:4))
        @test all(isapprox.( f_cdf_2.(x,1:4), cdf_rec_2.(x,1:4), atol=5e-2 ))

        @test_throws DomainError StochasticFluidQueues.left_point_mass(1,am,msh)
        tst = zeros(1,
            StochasticFluidQueues.n_phases(am)*StochasticFluidQueues.total_n_bases(msh) 
            + StochasticFluidQueues.N₋(am.S) + StochasticFluidQueues.N₊(am.S))
        tst[1] = 1.0
        @test StochasticFluidQueues.left_point_mass(2,am,msh)==tst
        
        @test_throws DomainError StochasticFluidQueues.right_point_mass(2,am,msh)
        tst2 = zeros(1,
            StochasticFluidQueues.n_phases(am)*StochasticFluidQueues.total_n_bases(msh) 
            + StochasticFluidQueues.N₋(am.S) + StochasticFluidQueues.N₊(am.S))
        tst2[end] = 1.0
        @test StochasticFluidQueues.right_point_mass(3,am,msh)==tst2

        # test point masses not at boundaries...
    end
end 

# test time 12_time_integration
# check 12_time_integration
@testset "time integration" begin
    
end

@testset "numerical checks" begin
    T = [-2.5 2 0.5; 1 -3 2; 1 2 -3]

    C = [0.0; 2.0; -6]
    m = [1;1;-1]
    S = StochasticFluidQueues.PhaseSet(C,m)

    bounds = [0.0,12]
    model = StochasticFluidQueues.FluidQueue(T,S,bounds)

    am = StochasticFluidQueues.augment_model(model)

    mtypes = (StochasticFluidQueues.DGMesh,
        StochasticFluidQueues.FVMesh,
        StochasticFluidQueues.FRAPMesh)
    for mtype in mtypes
        nodes = collect(0:0.1:12)
        order = 5
        msh = mtype(nodes,order)
        generator = StochasticFluidQueues.MakeFullGenerator(am,msh)

        b = zeros(1,size(generator,1))
        b[1] = 1.0
        if mtype==StochasticFluidQueues.FVMesh
            generator[:,1] = [
                ones(StochasticFluidQueues.N₋(am.S));
                repeat(StochasticFluidQueues.Δ(msh),StochasticFluidQueues.n_phases(am));
                ones(StochasticFluidQueues.N₊(am.S))]
        else
            generator[:,1] .= 1.0
        end

        stationary_coeffs = b/generator.B

        d = StochasticFluidQueues.SFMDistribution(stationary_coeffs,am,msh)

        stationary_cdf_estimate = StochasticFluidQueues.cdf(d,am)

        analytical_cdf = StochasticFluidQueues.StationaryDistributionX(am)[3]

        x_vec = bounds[1]:0.23:bounds[end]
        @testset "cdf points" begin
            for x in x_vec
                # display(x)
                @test analytical_cdf(x)≈stationary_cdf_estimate.(x,1:4) atol=1e-2
            end
        end
    end
end

# etc...
end

