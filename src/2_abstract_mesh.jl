"""

    Mesh 

Abstract type representing a mesh for a numerical scheme. 
"""
abstract type Mesh end 

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

    TotalNBases(mesh::Mesh)

Total number of bases in the stencil
"""
total_n_bases(mesh::Mesh) = n_bases(mesh) * n_intervals(mesh)

function MakeQBDidx(model::Model, mesh::Mesh)
    ## Make QBD index
    # N₊ = sum(model.C .>= 0)
    # N₋ = sum(model.C .<= 0)

    c = N₋(model.C)
    QBDidx = zeros(Int, NPhases(model) * TotalNBases(mesh) + N₊(model.C) + N₋(model.C))
    for k = 1:NIntervals(mesh), i = 1:NPhases(model), n = 1:NBases(mesh)
        c += 1
        QBDidx[c] = (i - 1) * TotalNBases(mesh) + (k - 1) * NBases(mesh) + n + N₋(model.C)
    end
    QBDidx[1:N₋] = 1:N₋
    QBDidx[(end-N₊(model.C)+1):end] = (NPhases(model) * TotalNBases(mesh) + N₋(model.C)) .+ (1:N₊(model.C))

    return QBDidx
end

export n_intervals, Δ, total_n_bases, MakeQBDidx, Mesh