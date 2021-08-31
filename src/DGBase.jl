"""
Constructs a DGMesh composite type, a subtype of the abstract type Mesh.

    DGMesh(
        model::SFFM.Model,
        Nodes::Array{Float64,1},
        NBases::Int;
        Fil::IndexDict=IndexDict(),
        Basis::String = "legendre",
    )

# Arguments
- `model`: a Model object
- `Nodes::Array{Float64,1}`: (K+1)×1 array, specifying the edges of the cells
- `NBases::Int`: specifying the number of bases within each cell (same for all
    cells)
- `Fil::IndexDict`: (optional) A dictionary of the sets Fᵢᵐ, they
    keys are Strings specifying i and m, i.e. `"2+"`, the values are BitArrays of
    boolean values which specify which cells of the stencil correspond to Fᵢᵐ. If no
    value specified then `Fil` is generated automatically evaluating ``r_i(x)`` at
    the modpoint of each cell.
- `Basis::String`: a string specifying whether to use the `"lagrange"` basis or
    the `"legendre"` basis

# Output
- a Mesh object with fieldnames:
    - `:NBases`: the number of bases in each cell
    - `:CellNodes`: Array of nodal points (cell edges + GLL points)
    - `:Fil`: As described in the arguments
    - `:Δ`:A vector of mesh widths, Δ[k] = x_{k+1} - x_k
    - `:NIntervals`: The number of cells
    - `:Nodes`: the cell edges
    - `:TotalNBases`: `NIntervals*NBases`
    - `:Basis`: a string specifying whether the

# Examples
TBC

#
A blank initialiser for a DGMesh.

    DGMesh()

Used for initialising a blank plot only. There is no reason to call this, ever. 
"""
struct DGMesh <: Mesh 
    Nodes::Array{<:Real,1}
    NBases::Int
    Fil::IndexDict
    Basis::String
end 
# Convenience constructors
function DGMesh(
    model::SFFM.Model,
    Nodes::Array{<:Real,1},
    NBases::Int;
    Fil::SFFM.IndexDict=SFFM.IndexDict(),
    Basis::String = "lagrange",
    v::Bool = false,
)
    if isempty(Fil)
        Fil = MakeFil(model, Nodes)
    end

    mesh = DGMesh(Nodes, NBases, Fil, Basis)

    v && println("UPDATE: DGMesh object created with fields ", fieldnames(SFFM.DGMesh))
    return mesh
end
function DGMesh()
    DGMesh(
        [0.0],
        0,
        IndexDict(),
        "",
    )
end

"""

    NBases(mesh::DGMesh)

Number of bases in a cell
"""
NBases(mesh::DGMesh) = mesh.NBases

"""

    NIntervals(mesh::Mesh)

Total number of cells for a mesh
"""
NIntervals(mesh::Mesh) = length(mesh.Nodes) - 1


"""

    Δ(mesh::Mesh)

The width of each cell
"""
Δ(mesh::Mesh) = mesh.Nodes[2:end] - mesh.Nodes[1:end-1]

"""

    TotalNBases(mesh::DGMesh)

Total number of bases in the stencil
"""
TotalNBases(mesh::Mesh) = NBases(mesh) * NIntervals(mesh)

"""

    CellNodes(mesh::DGMesh)

The positions of the GLJ nodes within each cell
"""
function CellNodes(mesh::DGMesh)
    meshNBases = NBases(mesh)
    meshNIntervals = NIntervals(mesh)
    cellNodes = zeros(Float64, NBases(mesh), NIntervals(mesh))
    if meshNBases > 1
        z = Jacobi.zglj(meshNBases, 0, 0) # the LGL nodes
    elseif meshNBases == 1
        z = 0.0
    end
    for i = 1:meshNIntervals
        # Map the LGL nodes on [-1,1] to each cell
        cellNodes[:, i] .= (mesh.Nodes[i+1] + mesh.Nodes[i]) / 2 .+ (mesh.Nodes[i+1] - mesh.Nodes[i]) / 2 * z
    end
    cellNodes[1,:] .+= sqrt(eps())
    if meshNBases>1
        cellNodes[end,:] .-= sqrt(eps())
    end
    return cellNodes
end

"""

    Basis(mesh::DGMesh)

Returns mesh.Basis; either "lagrange" or "legendre"
"""
Basis(mesh::DGMesh) = mesh.Basis


"""
Construct a generalised vandermonde matrix.

    vandermonde( nBases::Int)

Note: requires Jacobi package Pkg.add("Jacobi")

# Arguments
- `nBases::Int`: the degree of the basis

# Output
- a tuple with keys
    - `:V::Array{Float64,2}`: where `:V[:,i]` contains the values of the `i`th
        legendre polynomial evaluate at the GLL nodes.
    - `:inv`: the inverse of :V
    - `:D::Array{Float64,2}`: where `V.D[:,i]` contains the values of the derivative
        of the `i`th legendre polynomial evaluate at the GLL nodes.
"""
function vandermonde(nBases::Int)
    if nBases > 1
        z = Jacobi.zglj(nBases, 0, 0) # the LGL nodes
    elseif nBases == 1
        z = 0.0
    end
    V = zeros(Float64, nBases, nBases)
    DV = zeros(Float64, nBases, nBases)
    if nBases > 1
        for j = 1:nBases
            # compute the polynomials at gauss-labotto quadrature points
            V[:, j] = Jacobi.legendre.(z, j - 1) .* sqrt((2 * (j - 1) + 1) / 2)
            DV[:, j] = Jacobi.dlegendre.(z, j - 1) .* sqrt((2 * (j - 1) + 1) / 2)
        end
        # Compute the Gauss-Lobatto weights for numerical quadrature
        w =
            2.0 ./ (
                nBases *
                (nBases - 1) *
                Jacobi.legendre.(Jacobi.zglj(nBases, 0, 0), nBases - 1) .^ 2
            )
    elseif nBases == 1
        V .= [1/sqrt(2)]
        DV .= [0]
        w = [2]
    end
    return (V = V, inv = inv(V), D = DV, w = w)
end

function local_dg_operators(
    mesh::DGMesh;
    v::Bool = false,
)
    ## Construct local blocks
    V = vandermonde(NBases(mesh))
    if Basis(mesh) == "legendre"
        Dw = (
            DwInv = LinearAlgebra.diagm(0 => ones(Float64, NBases(mesh))),
            Dw = LinearAlgebra.diagm(0 => ones(Float64, NBases(mesh))),
        ) # function weights are not available for legendre basis as this is
        # in density land
        MLocal = Matrix{Float64}(LinearAlgebra.I(NBases(mesh)))
        GLocal = V.inv * V.D
        MInvLocal = Matrix{Float64}(LinearAlgebra.I(NBases(mesh)))
        Phi = V.V[[1; end], :]
    elseif Basis(mesh) == "lagrange"
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

    MakeLazyGenerator(
        model::SFFM.Model,
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
        `NPhases(model)*TotalNBases(mesh)×NPhases(model)*TotalNBases(mesh)`, the
        global approximation to `B`
    - `:QBDidx::Array{Int64,1}`: `NPhases(model)*TotalNBases(mesh)×1` vector of
        integers such such that `:B[QBDidx,QBDidx]` puts all the blocks relating
        to cell `k` next to each other
"""
function MakeLazyGenerator(
    model::SFFM.Model,
    mesh::DGMesh;
    v::Bool = false,
)

    m = local_dg_operators(mesh; v=v)
    blocks = (m.LowDiagBlock*m.MInv*2, (m.G+m.PosDiagBlock)*m.MInv*2, 
        -(m.G+m.NegDiagBlock)*m.MInv*2, m.UpDiagBlock*m.MInv*2)

    boundary_flux = (
        upper = (in = (m.Dw.DwInv * m.Phi[end, :]*2)[:], 
                out = (m.Phi[end, :]' * m.Dw.Dw * m.MInv)[:]),
        lower = (in = (m.Dw.DwInv * m.Phi[1, :]*2)[:], 
                out = (m.Phi[1, :]' * m.Dw.Dw * m.MInv)[:])
    )

    T = model.T
    C = model.C
    delta = Δ(mesh)
    D = LinearAlgebra.I(SFFM.NBases(mesh))
    pmidx = falses(size(T))

    out = SFFM.LazyGenerator(blocks,boundary_flux,T,C,delta,D,pmidx,mesh.Fil)
    v && println("UPDATE: LazyGenerator object created with keys ", keys(out))
    return out
end

"""
Uses Eulers method to integrate the matrix DE ``f'(x) = f(x)D`` to
approxiamte ``f(y)``.

    EulerDG(
        D::Union{Array{<:Real,2},SparseArrays.SparseMatrixCSC{Float64,Int64}},
        y::Real,
        x0::Array{<:Real};
        h::Float64 = 0.0001,
    )

# Arguments
- `D::Union{Array{<:Real,2},SparseArrays.SparseMatrixCSC{Float64,Int64}}`:
    the matrix ``D`` in the system of ODEs ``f'(x) = f(x)D``.
- `y::Real`: the value where we want to evaluate ``f(y)``.
- `x0::Array{<:Real}`: a row-vector initial condition.
- `h::Float64`: a stepsize for theEuler scheme.

# Output
- `f(y)::Array`: a row-vector approximation to ``f(y)``
"""
function EulerDG(
    D::AbstractArray{<:Real,2},
    y::Real,
    x0::AbstractArray{<:Real,2};
    h::Float64 = 0.0001,
)
    x = x0
    for t = h:h:y
        dx = h * (x * D)
        x = x + dx
    end
    return x
end


