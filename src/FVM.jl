# yuck. there were less fucks given making this
"""

    FVMesh(
        model::Model,
        nodes::Array{Float64,1};
        Fil::IndexDict=IndexDict(),
    ) 

Constructor for a mesh for a finite volume scheme. 
    Inputs: 
     - `model::Model` a Model object
     - `nodes::Array{Float64,1}` a vector specifying the cell edges
     - `Fil::IndexDict` an optional dictionary allocating the cells to the sets Fᵢᵐ
"""
struct FVMesh <: Mesh 
    nodes::Array{Float64,1}
    order::Int
    # Fil::IndexDict
end 

"""

    n_bases(mesh::FVMesh)

Constant 1
"""
n_bases(mesh::FVMesh) = 1
order(mesh::FVMesh) = mesh.order

"""

    cell_nodes(mesh::FVMesh)

The cell centres
"""
cell_nodes(mesh::FVMesh) = Array(((mesh.nodes[1:end-1] + mesh.nodes[2:end]) / 2 )')

"""

    basis(mesh::FVMesh)

Constant ""
"""
basis(mesh::FVMesh) = ""

total_n_bases(mesh::FVMesh) = n_intervals(mesh)

function MakeFVFlux(mesh::Mesh, order::Int)
    nNodes = total_n_bases(mesh)
    F = zeros(Float64,nNodes,nNodes)
    ptsLHS = Int(ceil(order/2))
    interiorCoeffs = lagrange_poly_basis(cell_nodes(mesh)[1:order],mesh.Nodes[ptsLHS+1])
    for n in 2:nNodes
        evalPt = mesh.Nodes[n]
        if n-ptsLHS-1 < 0
            nodesIdx = 1:order
            nodes = cell_nodes(mesh)[nodesIdx]
            coeffs = lagrange_poly_basis(nodes,evalPt)
        elseif n-ptsLHS-1+order > nNodes
            nodesIdx = (nNodes-order+1):nNodes
            nodes = cell_nodes(mesh)[nodesIdx]
            coeffs = lagrange_poly_basis(nodes,evalPt)
        else
            nodesIdx =  (n-ptsLHS-1) .+ (1:order)
            coeffs = interiorCoeffs
        end
        F[nodesIdx,n-1:n] += [-coeffs coeffs]./Δ(mesh)[n-1]
    end
    F[end-order+1:end,end] += -lagrange_poly_basis(cell_nodes(mesh)[end-order+1:end],mesh.nodes[end])./Δ(mesh)[end]
    return F
end

function MakeFullGenerator(model::Model, mesh::FVMesh; v::Bool=false)
    # N₊ = sum(model.C .>= 0)
    # N₋ = sum(model.C .<= 0)
    order = order(mesh)
    F = MakeFVFlux(mesh, order)

    B = SparseArrays.spzeros(
        Float64,
        n_phases(model) * n_intervals(mesh) + N₋(model.C) + N₊(model.C),
        n_phases(model) * n_intervals(mesh) + N₋(model.C) + N₊(model.C),
    )
    B[N₋(model.C)+1:end-N₊(model.C),N₋(model.C)+1:end-N₊(model.C)] = SparseArrays.kron(
            model.T,
            SparseArrays.I(n_intervals(mesh))
        )

    ## Make QBD index
    QBDidx = MakeQBDidx(model,mesh)
    
    # Boundary conditions
    T₋₋ = model.T[model.C.<=0,model.C.<=0]
    T₊₋ = model.T[model.C.>=0,:].*((model.C.<0)')
    T₋₊ = model.T[model.C.<=0,:].*((model.C.>0)')
    T₊₊ = model.T[model.C.>=0,model.C.>=0]
    # yuck
    begin 
        nodes = cell_nodes(mesh)[1:order]
        coeffs = lagrange_poly_basis(nodes,mesh.nodes[1])
        idxdown = ((1:order).+total_n_bases(mesh)*(findall(model.C .<= 0) .- 1)')[:] .+ N₋(model.C)
        B[idxdown, 1:N₋(model.C)] = LinearAlgebra.kron(
            LinearAlgebra.diagm(0 => model.C[model.C.<=0]),
            -coeffs,
        )
    end
    # inLower = [
    #     SparseArrays.diagm(abs.(model.C).*(model.C.<=0))[:,model.C.<=0]; 
    #     SparseArrays.zeros((n_intervals(mesh)-1)*n_phases(model),N₋)
    # ]
    outLower = [
        T₋₊./Δ(mesh)[1] SparseArrays.zeros(N₋(model.C),N₊(model.C)+(n_intervals(mesh)-1)*n_phases(model))
    ]
    begin
        nodes = cell_nodes(mesh)[end-order+1:end]
        coeffs = lagrange_poly_basis(nodes,mesh.nodes[end])
        idxup =
            ((1:order).+total_n_bases(mesh)*(findall(model.C .>= 0) .- 1)')[:] .+
            (N₋(model.C) + total_n_bases(mesh) - order)
        B[idxup, (end-N₊(model.C)+1):end] = LinearAlgebra.kron(
            LinearAlgebra.diagm(0 => model.C[model.C.>=0]),
            coeffs,
        )
    end
    # inUpper = [
    #     SparseArrays.zeros((n_intervals(mesh)-1)*n_phases(model),N₊);
    #     (SparseArrays.diagm(abs.(model.C).*(model.C.>=0)))[:,model.C.>=0]
    # ]
    outUpper = [
        SparseArrays.zeros(N₊(model.C),N₋(model.C)+(n_intervals(mesh)-1)*n_phases(model)) T₊₋./Δ(mesh)[end]
    ]
    
    B[1:N₋(model.C),QBDidx] = [T₋₋ outLower]
    B[end-N₊(model.C)+1:end,QBDidx] = [outUpper T₊₊]
    # B[QBDidx[N₋+1:end-N₊],1:N₋] = inLower
    # B[QBDidx[N₋+1:end-N₊],(end-N₊+1):end] = inUpper
    for i = 1:n_phases(model)
        idx = ((i-1)*n_intervals(mesh)+1:i*n_intervals(mesh)) .+ N₋(model.C)
        if model.C[i] > 0
            B[idx, idx] += model.C[i] * F
        elseif model.C[i] < 0
            B[idx, idx] += abs(model.C[i]) * F[end:-1:1,end:-1:1]
        end
    end

    # BDict = MakeDict(B,model,mesh)
    out = FullGenerator(B, mesh.Fil)
    v && println("FullGenerator created with keys ", keys(out))
    return out
end