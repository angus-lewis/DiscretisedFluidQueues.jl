# yuck. there were less fucks given making this
"""
    FVMesh{T} <: Mesh{T}

A finite volume discretisation scheme for a DiscretisedFluidQueue. 

# Arguments
 - `nodes`: the cell edges (the nodes are at the center of the cell)
 - `order`: the order of the polynomial interpolation used to approximate the flux at the 
 cell edges.
"""
struct FVMesh{T} <: Mesh{T}
    nodes::T
    order::Int
    function FVMesh{T}(nodes::T,order::Int) where T
        (order<=0)&&throw(DomainError("order must be positive"))
        (nodes[1]!=0.0)&&throw(DomainError("first node must be 0"))
        return new{T}(nodes,order)
    end
end 
# Convenience constructors
FVMesh(nodes::T,order::Int) where T = FVMesh{T}(nodes,order)

"""

    n_bases_per_cell(mesh::FVMesh)

Constant=1
"""
n_bases_per_cell(mesh::FVMesh) = 1
_order(mesh::FVMesh) = mesh.order

"""

    cell_nodes(mesh::FVMesh)

The cell centres
"""
cell_nodes(mesh::FVMesh) = Array(((mesh.nodes[1:end-1] + mesh.nodes[2:end]) / 2 )')

n_bases_per_phase(mesh::FVMesh) = n_intervals(mesh)

function MakeFVFlux(mesh::Mesh)
    order = _order(mesh)
    nNodes = n_bases_per_phase(mesh)
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


function build_full_generator(dq::DiscretisedFluidQueue{FVMesh{T}}; v::Bool=false) where T
    model = dq.model
    
    order = _order(dq.mesh)
    F = MakeFVFlux(dq.mesh)

    C = rates(model.S)
    n₋ = N₋(model.S)
    n₊ = N₊(model.S)

    B = SparseArrays.spzeros(
        Float64,
        n_phases(model) * n_intervals(dq) + n₋ + n₊,
        n_phases(model) * n_intervals(dq) + n₋ + n₊,
    )
    B[n₋+1:end-n₊,n₋+1:end-n₊] = SparseArrays.kron(
            model.T,
            SparseArrays.I(n_intervals(dq))
        )

    ## Make QBD index
    QBDidx = qbd_idx(dq)
    
    # Boundary conditions
    T₋₋ = model.T[_has_left_boundary.(model.S),_has_left_boundary.(model.S)]
    T₊₋ = model.T[_has_right_boundary.(model.S),:].*((C.<0)')
    T₋₊ = model.T[_has_left_boundary.(model.S),:].*((C.>0)')
    T₊₊ = model.T[_has_right_boundary.(model.S),_has_right_boundary.(model.S)]
    # yuck
    begin 
        nodes = cell_nodes(dq)[1:order]
        coeffs = lagrange_polynomials(nodes,dq.mesh.nodes[1])
        idxdown = ((1:order).+n_bases_per_phase(dq)*(findall(_has_left_boundary.(model.S)) .- 1)')[:] .+ n₋
        down_rates = LinearAlgebra.diagm(0 => C[_has_left_boundary.(model.S)])
        B[idxdown, 1:n₋] = LinearAlgebra.kron(down_rates,-coeffs)
    end
    outLower = [
        T₋₊./Δ(dq,1) SparseArrays.zeros(n₋,n₊+(n_intervals(dq)-1)*n_phases(model))
    ]
    begin
        nodes = cell_nodes(dq)[end-order+1:end]
        coeffs = lagrange_polynomials(nodes,dq.mesh.nodes[end])
        idxup =
            ((1:order).+n_bases_per_phase(dq)*(findall(_has_right_boundary.(model.S)) .- 1)')[:] .+
            (n₋ + n_bases_per_phase(dq) - order)
        B[idxup, (end-n₊+1):end] = LinearAlgebra.kron(
            LinearAlgebra.diagm(0 => C[_has_right_boundary.(model.S)]),
            coeffs,
        )
    end
    outUpper = [
        SparseArrays.zeros(n₊,n₋+(n_intervals(dq)-1)*n_phases(model)) T₊₋./Δ(dq,n_intervals(dq))
    ]
    
    B[1:n₋,QBDidx] = [T₋₋ outLower]
    B[end-n₊+1:end,QBDidx] = [outUpper T₊₊]
    for i = 1:n_phases(model)
        idx = ((i-1)*n_intervals(dq)+1:i*n_intervals(dq)) .+ n₋
        if C[i] > 0
            B[idx, idx] += C[i] * F
        elseif C[i] < 0
            B[idx, idx] += abs(C[i]) * F[end:-1:1,end:-1:1]
        end
    end

    out = FullGenerator(B[QBDidx,QBDidx],dq)#, mesh.Fil)
    v && println("FullGenerator created with keys ", keys(out))
    return out
end