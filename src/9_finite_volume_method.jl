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
_order(mesh::FVMesh) = mesh.order

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

function MakeFVFlux(mesh::Mesh)
    order = _order(mesh)
    nNodes = total_n_bases(mesh)
    F = zeros(Float64,nNodes,nNodes)
    ptsLHS = Int(ceil(order/2))
    interiorCoeffs = lagrange_polynomials(cell_nodes(mesh)[1:order],mesh.nodes[ptsLHS+1])
    for n in 2:nNodes
        evalPt = mesh.nodes[n]
        if n-ptsLHS-1 < 0
            nodesIdx = 1:order
            nodes = cell_nodes(mesh)[nodesIdx]
            coeffs = lagrange_polynomials(nodes,evalPt)
        elseif n-ptsLHS-1+order > nNodes
            nodesIdx = (nNodes-order+1):nNodes
            nodes = cell_nodes(mesh)[nodesIdx]
            coeffs = lagrange_polynomials(nodes,evalPt)
        else
            nodesIdx =  (n-ptsLHS-1) .+ (1:order)
            coeffs = interiorCoeffs
        end
        F[nodesIdx,n-1:n] += [-coeffs coeffs]./Δ(mesh)[n-1]
    end
    F[end-order+1:end,end] += -lagrange_polynomials(cell_nodes(mesh)[end-order+1:end],mesh.nodes[end])./Δ(mesh)[end]
    return F
end

function MakeFullGenerator(model::Model, mesh::FVMesh; v::Bool=false)
    # N₊ = sum(model.C .>= 0)
    # N₋ = sum(model.C .<= 0)
    order = _order(mesh)
    F = MakeFVFlux(mesh)

    m = membership(model.S)
    C = rates(model.S)
    n₋ = N₋(model.S)
    n₊ = N₊(model.S)

    B = SparseArrays.spzeros(
        Float64,
        n_phases(model) * n_intervals(mesh) + n₋ + n₊,
        n_phases(model) * n_intervals(mesh) + n₋ + n₊,
    )
    B[n₋+1:end-n₊,n₋+1:end-n₊] = SparseArrays.kron(
            model.T,
            SparseArrays.I(n_intervals(mesh))
        )

    ## Make QBD index
    QBDidx = MakeQBDidx(model,mesh)
    
    # Boundary conditions
    T₋₋ = model.T[_has_left_boundary.(m),_has_left_boundary.(m)]
    T₊₋ = model.T[_has_right_boundary.(m),:].*((C.<0)')
    T₋₊ = model.T[_has_left_boundary.(m),:].*((C.>0)')
    T₊₊ = model.T[_has_right_boundary.(m),_has_right_boundary.(m)]
    # yuck
    begin 
        nodes = cell_nodes(mesh)[1:order]
        coeffs = lagrange_polynomials(nodes,mesh.nodes[1])
        idxdown = ((1:order).+total_n_bases(mesh)*(findall(_has_left_boundary.(m)) .- 1)')[:] .+ n₋
        down_rates = LinearAlgebra.diagm(0 => C[_has_left_boundary.(m)])
        B[idxdown, 1:n₋] = LinearAlgebra.kron(down_rates,-coeffs)
    end
    # inLower = [
    #     SparseArrays.diagm(abs.(model.C).*(model.C.<=0))[:,model.C.<=0]; 
    #     SparseArrays.zeros((n_intervals(mesh)-1)*n_phases(model),N₋)
    # ]
    outLower = [
        T₋₊./Δ(mesh,1) SparseArrays.zeros(n₋,n₊+(n_intervals(mesh)-1)*n_phases(model))
    ]
    begin
        nodes = cell_nodes(mesh)[end-order+1:end]
        coeffs = lagrange_polynomials(nodes,mesh.nodes[end])
        idxup =
            ((1:order).+total_n_bases(mesh)*(findall(_has_right_boundary.(m)) .- 1)')[:] .+
            (n₋ + total_n_bases(mesh) - order)
        B[idxup, (end-n₊+1):end] = LinearAlgebra.kron(
            LinearAlgebra.diagm(0 => C[_has_right_boundary.(m)]),
            coeffs,
        )
    end
    # inUpper = [
    #     SparseArrays.zeros((n_intervals(mesh)-1)*n_phases(model),N₊);
    #     (SparseArrays.diagm(abs.(model.C).*(model.C.>=0)))[:,model.C.>=0]
    # ]
    outUpper = [
        SparseArrays.zeros(n₊,n₋+(n_intervals(mesh)-1)*n_phases(model)) T₊₋./Δ(mesh,n_intervals(mesh))
    ]
    
    B[1:n₋,QBDidx] = [T₋₋ outLower]
    B[end-n₊+1:end,QBDidx] = [outUpper T₊₊]
    # B[QBDidx[N₋+1:end-N₊],1:N₋] = inLower
    # B[QBDidx[N₋+1:end-N₊],(end-N₊+1):end] = inUpper
    for i = 1:n_phases(model)
        idx = ((i-1)*n_intervals(mesh)+1:i*n_intervals(mesh)) .+ n₋
        if C[i] > 0
            B[idx, idx] += C[i] * F
        elseif C[i] < 0
            B[idx, idx] += abs(C[i]) * F[end:-1:1,end:-1:1]
        end
    end

    # BDict = MakeDict(B,model,mesh)
    out = FullGenerator(B)#, mesh.Fil)
    v && println("FullGenerator created with keys ", keys(out))
    return out
end