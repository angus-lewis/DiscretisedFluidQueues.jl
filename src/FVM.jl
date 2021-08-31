# yuck. there were less fucks given making this
"""

    FVMesh(
        model::SFFM.Model,
        Nodes::Array{Float64,1};
        Fil::IndexDict=IndexDict(),
    ) 

Constructor for a mesh for a finite volume scheme. 
    Inputs: 
     - `model::Model` a Model object
     - `Nodes::Array{Float64,1}` a vector specifying the cell edges
     - `Fil::IndexDict` an optional dictionary allocating the cells to the sets Fᵢᵐ
"""
struct FVMesh <: SFFM.Mesh 
    Nodes::Array{Float64,1}
    order::Int
    Fil::IndexDict
end 
function FVMesh(
    model::SFFM.Model,
    Nodes::Array{Float64,1},
    order::Int;
    Fil::IndexDict=IndexDict(),
) 
    ## Construct the sets Fᵐ = ⋃ᵢ Fᵢᵐ, global index for sets of type m
    if isempty(Fil)
        Fil = MakeFil(model, Nodes)
    end

    return FVMesh(Nodes, order, Fil)
end


"""

    NBases(mesh::FVMesh)

Constant 1
"""
NBases(mesh::FVMesh) = 1
Order(mesh::FVMesh) = mesh.order

"""

    CellNodes(mesh::FVMesh)

The cell centres
"""
CellNodes(mesh::FVMesh) = Array(((mesh.Nodes[1:end-1] + mesh.Nodes[2:end]) / 2 )')

"""

    Basis(mesh::FVMesh)

Constant ""
"""
Basis(mesh::FVMesh) = ""

TotalNBases(mesh::FVMesh) = NIntervals(mesh)

function MakeFVFlux(mesh::SFFM.Mesh, order::Int)
    nNodes = TotalNBases(mesh)
    F = zeros(Float64,nNodes,nNodes)
    ptsLHS = Int(ceil(order/2))
    interiorCoeffs = lagrange_poly_basis(CellNodes(mesh)[1:order],mesh.Nodes[ptsLHS+1])
    for n in 2:nNodes
        evalPt = mesh.Nodes[n]
        if n-ptsLHS-1 < 0
            nodesIdx = 1:order
            nodes = CellNodes(mesh)[nodesIdx]
            coeffs = lagrange_poly_basis(nodes,evalPt)
        elseif n-ptsLHS-1+order > nNodes
            nodesIdx = (nNodes-order+1):nNodes
            nodes = CellNodes(mesh)[nodesIdx]
            coeffs = lagrange_poly_basis(nodes,evalPt)
        else
            nodesIdx =  (n-ptsLHS-1) .+ (1:order)
            coeffs = interiorCoeffs
        end
        F[nodesIdx,n-1:n] += [-coeffs coeffs]./Δ(mesh)[n-1]
    end
    F[end-order+1:end,end] += -lagrange_poly_basis(CellNodes(mesh)[end-order+1:end],mesh.Nodes[end])./Δ(mesh)[end]
    return F
end

function MakeFullGenerator(model::SFFM.Model, mesh::SFFM.FVMesh; v::Bool=false)
    N₊ = sum(model.C .>= 0)
    N₋ = sum(model.C .<= 0)
    order = Order(mesh)
    F = SFFM.MakeFVFlux(mesh, order)

    B = SparseArrays.spzeros(
        Float64,
        NPhases(model) * NIntervals(mesh) + N₋ + N₊,
        NPhases(model) * NIntervals(mesh) + N₋ + N₊,
    )
    B[N₋+1:end-N₊,N₋+1:end-N₊] = SparseArrays.kron(
            model.T,
            SparseArrays.I(NIntervals(mesh))
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
        nodes = CellNodes(mesh)[1:order]
        coeffs = lagrange_poly_basis(nodes,mesh.Nodes[1])
        idxdown = ((1:order).+TotalNBases(mesh)*(findall(model.C .<= 0) .- 1)')[:] .+ N₋
        B[idxdown, 1:N₋] = LinearAlgebra.kron(
            LinearAlgebra.diagm(0 => model.C[model.C.<=0]),
            -coeffs,
        )
    end
    # inLower = [
    #     SparseArrays.diagm(abs.(model.C).*(model.C.<=0))[:,model.C.<=0]; 
    #     SparseArrays.zeros((NIntervals(mesh)-1)*NPhases(model),N₋)
    # ]
    outLower = [
        T₋₊./Δ(mesh)[1] SparseArrays.zeros(N₋,N₊+(NIntervals(mesh)-1)*NPhases(model))
    ]
    begin
        nodes = CellNodes(mesh)[end-order+1:end]
        coeffs = lagrange_poly_basis(nodes,mesh.Nodes[end])
        idxup =
            ((1:order).+TotalNBases(mesh)*(findall(model.C .>= 0) .- 1)')[:] .+
            (N₋ + TotalNBases(mesh) - order)
        B[idxup, (end-N₊+1):end] = LinearAlgebra.kron(
            LinearAlgebra.diagm(0 => model.C[model.C.>=0]),
            coeffs,
        )
    end
    # inUpper = [
    #     SparseArrays.zeros((NIntervals(mesh)-1)*NPhases(model),N₊);
    #     (SparseArrays.diagm(abs.(model.C).*(model.C.>=0)))[:,model.C.>=0]
    # ]
    outUpper = [
        SparseArrays.zeros(N₊,N₋+(NIntervals(mesh)-1)*NPhases(model)) T₊₋./Δ(mesh)[end]
    ]
    
    B[1:N₋,QBDidx] = [T₋₋ outLower]
    B[end-N₊+1:end,QBDidx] = [outUpper T₊₊]
    # B[QBDidx[N₋+1:end-N₊],1:N₋] = inLower
    # B[QBDidx[N₋+1:end-N₊],(end-N₊+1):end] = inUpper
    for i = 1:NPhases(model)
        idx = ((i-1)*NIntervals(mesh)+1:i*NIntervals(mesh)) .+ N₋
        if model.C[i] > 0
            B[idx, idx] += model.C[i] * F
        elseif model.C[i] < 0
            B[idx, idx] += abs(model.C[i]) * F[end:-1:1,end:-1:1]
        end
    end

    BDict = SFFM.MakeDict(B,model,mesh)
    out = FullGenerator(BDict, B, mesh.Fil)
    v && println("FullGenerator created with keys ", keys(out))
    return out
end