@testset "Mesh Basics" begin 
    @testset "DG Mesh basics" begin    
        @test typeof(dgmesh)==DGMesh
        @test typeof(dgmesh)<:Mesh
        @test n_intervals(dgmesh)==length(nodes)-1
        @test Δ(dgmesh) == nodes[2:end]-nodes[1:end-1]
        @test Δ(dgmesh,1) == nodes[2]-nodes[1]
        @test n_bases_per_phase(dgmesh) == (length(nodes)-1)*nbases
        @test n_bases_per_cell(dgmesh) == 3
        @test cell_nodes(dgmesh)≈
            [nodes[1:end-1]';(nodes[1:end-1]'+nodes[2:end]')/2;nodes[2:end]'] atol=1e-5
        @test DiscretisedFluidQueues.basis(dgmesh) == "lagrange"
        # @test local_dg_operators(dgmesh) == ???
        # test QBDidx 
    end

    @testset "FV Mesh basics" begin    
        @test typeof(fvmesh)==FVMesh
        @test typeof(fvmesh)<:Mesh
        @test n_intervals(fvmesh)==length(nodes)-1
        @test Δ(fvmesh) == nodes[2:end]-nodes[1:end-1]
        @test Δ(fvmesh,1) == nodes[2]-nodes[1]
        @test n_bases_per_phase(fvmesh) == (length(nodes)-1)
        @test n_bases_per_cell(fvmesh) == 1
        @test DiscretisedFluidQueues._order(fvmesh) == fv_order
        @test cell_nodes(fvmesh)≈Array(((fvmesh.nodes[1:end-1] + fvmesh.nodes[2:end]) / 2 )') atol=1e-5
        @test DiscretisedFluidQueues.basis(fvmesh) == ""
    end

    @testset "FRAP Mesh basics" begin    
        @test typeof(frapmesh)==FRAPMesh
        @test typeof(frapmesh)<:Mesh
        @test n_intervals(frapmesh)==length(nodes)-1
        @test Δ(frapmesh) == nodes[2:end]-nodes[1:end-1]
        @test Δ(frapmesh,1) == nodes[2]-nodes[1]
        @test n_bases_per_phase(frapmesh) == (length(nodes)-1)*order
        @test n_bases_per_cell(frapmesh) == order
        @test cell_nodes(frapmesh)≈Array(((frapmesh.nodes[1:end-1] + frapmesh.nodes[2:end]) / 2 )') atol=1e-5
        @test DiscretisedFluidQueues.basis(frapmesh) == ""
    end
end 