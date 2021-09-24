"""
Constructs a DGMesh composite type, a subtype of the abstract type Mesh.

    DGMesh(
        model::Model,
        nodes::Array{Float64,1},
        n_bases::Int;
        Fil::IndexDict=IndexDict(),
        basis::String = "legendre",
    )

# Arguments
- `model`: a Model object
- `nodes::Array{Float64,1}`: (K+1)×1 array, specifying the edges of the cells
- `n_bases::Int`: specifying the number of bases within each cell (same for all
    cells)
- `Fil::IndexDict`: (optional) A dictionary of the sets Fᵢᵐ, they
    keys are Strings specifying i and m, i.e. `"2+"`, the values are BitArrays of
    boolean values which specify which cells of the stencil correspond to Fᵢᵐ. If no
    value specified then `Fil` is generated automatically evaluating ``r_i(x)`` at
    the modpoint of each cell.
- `basis::String`: a string specifying whether to use the `"lagrange"` basis or
    the `"legendre"` basis

# Output
- a Mesh object with fieldnames:
    - `:n_bases`: the number of bases in each cell
    - `:cell_nodes`: Array of nodal points (cell edges + GLL points)
    - `:Fil`: As described in the arguments
    - `:Δ`:A vector of mesh widths, Δ[k] = x_{k+1} - x_k
    - `:NIntervals`: The number of cells
    - `:nodes`: the cell edges
    - `:n_bases_per_phase`: `NIntervals*n_bases`
    - `:basis`: a string specifying whether the

# Examples
TBC

#
A blank initialiser for a DGMesh.

    DGMesh()

Used for initialising a blank plot only. There is no reason to call this, ever. 
"""
struct DGMesh <: Mesh 
    nodes::Array{<:Real,1}
    n_bases::Int
    basis::String
end 
# Convenience constructors
function DGMesh(nodes::Array{<:Real,1},n_bases::Int)
    basis = "lagrange"
    mesh = DGMesh(nodes, n_bases,basis)
    return mesh
end
DGMesh() = DGMesh([0.0],0,"")

"""

    n_bases_per_cell(mesh::DGMesh)

Number of bases in a cell
"""
n_bases_per_cell(mesh::DGMesh) = mesh.n_bases


"""

    cell_nodes(mesh::DGMesh)

The positions of the GLJ nodes within each cell
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

    basis(mesh::DGMesh)

Returns mesh.basis; either "lagrange" or "legendre"
"""
basis(mesh::DGMesh) = mesh.basis

function local_dg_operators(
    mesh::DGMesh;
    v::Bool = false,
)
    ## Construct local blocks
    V = vandermonde(n_bases_per_cell(mesh))
    if basis(mesh) == "legendre"
        Dw = (
            DwInv = LinearAlgebra.diagm(0 => ones(Float64, n_bases_per_cell(mesh))),
            Dw = LinearAlgebra.diagm(0 => ones(Float64, n_bases_per_cell(mesh))),
        ) # function weights are not available for legendre basis as this is
        # in density land
        MLocal = Matrix{Float64}(LinearAlgebra.I(n_bases_per_cell(mesh)))
        GLocal = V.inv * V.D
        MInvLocal = Matrix{Float64}(LinearAlgebra.I(n_bases_per_cell(mesh)))
        Phi = V.V[[1; end], :]
    elseif basis(mesh) == "lagrange"
        Dw = (
            DwInv = LinearAlgebra.diagm(0 => 1.0 ./ V.w),
            Dw = LinearAlgebra.diagm(0 => V.w),
        )# function weights so that we can work in probability land as
        # opposed to density land

        MLocal = Dw.DwInv * V.inv' * V.inv * Dw.Dw
        GLocal = Dw.DwInv * V.inv' * V.inv * (V.D * V.inv) * Dw.Dw
        MInvLocal = Dw.DwInv * V.V * V.V' * Dw.Dw
        Phi = (V.inv*V.V)[[1; end], :]
    end

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

"""
Creates the DG approximation to the generator `B`.

    build_lazy_generator(
        model::Model,
        mesh::DGMesh,
        Matrices::NamedTuple;
    )

# Arguments
- `model`: A Model object
- `mesh`: A Mesh object
- `Matrices`: A Matrices tuple from `MakeMatrices`

# Output
- A Generator with fields `:BDict, :B, :QBDidx`
    - `:BDict::Dict{String,Array{Float64,2}}`: a dictionary storing Bᵢⱼˡᵐ with
        keys string(i,j,ℓ,m), and values Bᵢⱼˡᵐ, i.e. `B.BDict["12+-"]` = B₁₂⁺⁻
    - `:B::SparseArrays.SparseMatrixCSC{Float64,Int64}`:
        `NPhases(model)*n_bases_per_phase(mesh)×NPhases(model)*n_bases_per_phase(mesh)`, the
        global approximation to `B`
    - `:QBDidx::Array{Int64,1}`: `NPhases(model)*n_bases_per_phase(mesh)×1` vector of
        integers such such that `:B[QBDidx,QBDidx]` puts all the blocks relating
        to cell `k` next to each other
"""
function build_lazy_generator(dq::DiscretisedFluidQueue{DGMesh}; v::Bool = false)

    m = local_dg_operators(dq.mesh; v=v)
    blocks = (m.LowDiagBlock*m.MInv*2, (m.G+m.PosDiagBlock)*m.MInv*2, 
        -(m.G+m.NegDiagBlock)*m.MInv*2, m.UpDiagBlock*m.MInv*2)

    boundary_flux = (
        upper = (in = (m.Dw.DwInv * m.Phi[end, :]*2)[:], 
                out = (m.Phi[end, :]' * m.Dw.Dw * m.MInv)[:]),
        lower = (in = (m.Dw.DwInv * m.Phi[1, :]*2)[:], 
                out = (m.Phi[1, :]' * m.Dw.Dw * m.MInv)[:])
    )

    D = LinearAlgebra.I(n_bases_per_cell(dq))

    out = LazyGenerator(dq,blocks,boundary_flux,D)
    v && println("UPDATE: LazyGenerator object created with keys ", keys(out))
    return out
end

