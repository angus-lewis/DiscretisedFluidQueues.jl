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
total_n_bases(mesh::Mesh) = n_bases(mesh) * n+intervals(mesh)