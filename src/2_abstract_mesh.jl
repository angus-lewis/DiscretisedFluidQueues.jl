"""

    Mesh 

Abstract type representing a discretisation mesh for a numerical scheme. 
"""
abstract type Mesh end 

"""

    n_intervals(m)

Total number of cells for a Mesh or DiscretisedFluidQueue.
"""
n_intervals(mesh::Mesh) = length(mesh.nodes) - 1

"""

    Δ(m)

The width of each cell for a Mesh or DiscretisedFluidQueue.
"""
Δ(mesh::Mesh) = mesh.nodes[2:end] - mesh.nodes[1:end-1]
"""

    Δ(m,k::Int)

The width of cell k for a Mesh or DiscretisedFluidQueue.
"""
Δ(mesh::Mesh,k) = mesh.nodes[k+1] - mesh.nodes[k]

"""

    n_bases_per_phase(m)

Total number of basis functions used to represent the fluid queue for each phase 
    of a Mesh or DiscretisedFluidQueue.
"""
n_bases_per_phase(mesh::Mesh) = n_bases_per_cell(mesh) * n_intervals(mesh)
