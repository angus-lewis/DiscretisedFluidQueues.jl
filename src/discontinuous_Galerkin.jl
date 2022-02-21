"""
    DGMesh{T} <: Mesh{T}

A structure representing a discretisation scheme to be used for a DiscretisedFluidQueue. 

# Arguments:
- `nodes::AbstractArray{Float64, 1}`: The edges of the cells.
- `n_bases::Int`: The number of basis functions used to represent the solution on each cell
"""
struct DGMesh{T} <: Mesh{T}
    nodes::T
    n_bases::Int
    function DGMesh{T}(nodes::T,n_bases::Int) where T
        (n_bases<=0)&&throw(DomainError("n_bases must be positive"))
        (nodes[1]!=0.0)&&throw(DomainError("first node must be 0"))
        return new{T}(nodes,n_bases)
    end
end 
# Convenience constructors
DGMesh(nodes::T,n_bases::Int) where T = DGMesh{T}(nodes::T,n_bases::Int) 
DGMesh() = DGMesh([0.0],0)

"""

    n_bases_per_cell(mesh::DGMesh)

Number of bases in a cell
"""
n_bases_per_cell(mesh::DGMesh) = mesh.n_bases


"""

    cell_nodes(mesh::DGMesh)

The positions of the GLJ nodes within each cell of a mesh.
"""
function cell_nodes(mesh::DGMesh)
    cellnodes = zeros(Float64, n_bases_per_cell(mesh), n_intervals(mesh))
    if n_bases_per_cell(mesh) > 1
        z = Jacobi.zglj(n_bases_per_cell(mesh), 0, 0) # the LGL nodes
    elseif n_bases_per_cell(mesh) == 1
        z = 0.0
    end
    for i = 1:n_intervals(mesh)
        # Map the LGL nodes on [-1,1] to each cell
        cellnodes[:, i] .= (mesh.nodes[i+1] + mesh.nodes[i]) / 2 .+ (mesh.nodes[i+1] - mesh.nodes[i]) / 2 * z
    end
    cellnodes[1,:] .+= sqrt(eps()) # move the cell edges because funny things can happen at boundaries
    if n_bases_per_cell(mesh)>1
        cellnodes[end,:] .-= sqrt(eps())
    end
    return cellnodes
end

"""
    local_dg_operators(mesh::DGMesh; v::Bool = false)

Construct the block matrices and vectors which define the discretised
generator (i.e. the `blocks`, and `boundary_flux` in `LazyGenerator`).
"""
function local_dg_operators(
    mesh::DGMesh;
    v::Bool = false,
)
    ## Construct local blocks
    V = vandermonde(n_bases_per_cell(mesh))

    Dw = (
        DwInv = LinearAlgebra.diagm(0 => 1.0 ./ V.w),
        Dw = LinearAlgebra.diagm(0 => V.w),
    )# function weights so that we can work in probability land as
    # opposed to density land

    MLocal = Dw.DwInv * V.inv' * V.inv * Dw.Dw
    GLocal = Dw.DwInv * V.inv' * V.inv * (V.D * V.inv) * Dw.Dw
    MInvLocal = Dw.DwInv * V.V * V.V' * Dw.Dw
    Phi = (V.inv*V.V)[[1; end], :]

    PosDiagBlock = -Dw.DwInv * Phi[end, :] * Phi[end, :]' * Dw.Dw
    NegDiagBlock = Dw.DwInv * Phi[1, :] * Phi[1, :]' * Dw.Dw
    UpDiagBlock = Dw.DwInv * Phi[end, :] * Phi[1, :]' * Dw.Dw
    LowDiagBlock = Dw.DwInv * Phi[1, :] * Phi[end, :]' * Dw.Dw

    out = (
        G = GLocal, M = MLocal, MInv = MInvLocal, V = V, Phi = Phi, Dw = Dw, 
        PosDiagBlock = PosDiagBlock, NegDiagBlock = NegDiagBlock,
        UpDiagBlock = UpDiagBlock, LowDiagBlock = LowDiagBlock,
        )
    v && println("UPDATE: Local operators created; ", keys(out))
    return out 
end

function build_lazy_generator(dq::DiscretisedFluidQueue{DGMesh{T}}; v::Bool = false) where T

    m = local_dg_operators(dq.mesh; v=v)
    blocks = (m.LowDiagBlock*m.MInv*2, (m.G+m.PosDiagBlock)*m.MInv*2, 
        -(m.G+m.NegDiagBlock)*m.MInv*2, m.UpDiagBlock*m.MInv*2)

    boundary_flux = BoundaryFlux(
        OneBoundaryFlux((m.Dw.DwInv * m.Phi[end, :]*2)[:],
            (m.Phi[end, :]' * m.Dw.Dw * m.MInv)[:]),
        OneBoundaryFlux((m.Dw.DwInv * m.Phi[1, :]*2)[:], 
            (m.Phi[1, :]' * m.Dw.Dw * m.MInv)[:])
    )

    D = Matrix{Float64}(LinearAlgebra.I(n_bases_per_cell(dq)))

    out = LazyGenerator(dq,blocks,boundary_flux,D)
    v && println("UPDATE: LazyGenerator object created with keys ", keys(out))
    return out
end
