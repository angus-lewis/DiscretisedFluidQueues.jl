"""

    Mesh 

Abstract type representing a discretisation mesh for a numerical scheme. 
"""
abstract type Mesh end 

"""
 
"""
struct DiscretisedFluidQueue{T<:Mesh}
    model::Model
    mesh::T
end

"""

    n_intervals(mesh::Mesh)

Total number of cells for a mesh. All Mesh objects must have field nodes 
    specifying the cell edges
"""
n_intervals(mesh::Mesh) = length(mesh.nodes) - 1

"""

    Δ(mesh::Mesh)

The width of each cell
"""
Δ(mesh::Mesh) = mesh.nodes[2:end] - mesh.nodes[1:end-1]
"""

    Δ(mesh::Mesh,k::Int)

The width of cell k
"""
Δ(mesh::Mesh,k) = mesh.nodes[k+1] - mesh.nodes[k]

"""

    total_n_bases(mesh::Mesh)

Total number of bases in the stencil
"""
total_n_bases(mesh::Mesh) = n_bases(mesh) * n_intervals(mesh)

function MakeQBDidx(dq::DiscretisedFluidQueue)
    ## Make QBD index
    model = dq.model
    mesh = dq.mesh

    c = N₋(model.S)
    n₊ = N₊(model.S)
    n₋ = N₋(model.S)
    QBDidx = zeros(Int, n_phases(model) * total_n_bases(mesh) + n₊ + n₋)
    for k = 1:n_intervals(mesh), i = 1:n_phases(model), n = 1:n_bases(mesh)
        c += 1
        QBDidx[c] = (i - 1) * total_n_bases(mesh) + (k - 1) * n_bases(mesh) + n + n₋
    end
    QBDidx[1:n₋] = 1:n₋
    QBDidx[(end-n₊+1):end] = (n_phases(model) * total_n_bases(mesh) + n₋) .+ (1:n₊)

    return QBDidx
end

export n_intervals, Δ, total_n_bases, MakeQBDidx, Mesh