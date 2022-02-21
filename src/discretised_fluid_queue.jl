"""
    DiscretisedFluidQueue{T <: Mesh}

A container to hold a BoundedFluidQueue and a discretisation scheme (a mesh). 
Parametric type depending on the type of discretisation scheme (<:Mesh)
# Arguments:
- `model::Model`: a BoundedFluidQueue
- `mesh::T`: either a DGMesh, FVMesh or FRAPMesh specifying the discretisation scheme for the 
    continuous level of the fluid queue, X(t). 
"""
struct DiscretisedFluidQueue{T<:Mesh}
    model::Model
    mesh::T
    function DiscretisedFluidQueue{T}(model::Model,mesh::T) where T
        (model.b!=mesh.nodes[end])&&throw(DomainError("mesh.nodes[end] must equal model.b"))
        return new{T}(model::Model,mesh::T)
    end
end
DiscretisedFluidQueue(model::Model,mesh::T) where T = 
    DiscretisedFluidQueue{T}(model,mesh)

n_phases(dq::DiscretisedFluidQueue) = n_phases(dq.model)
rates(dq::DiscretisedFluidQueue,i::Int) = rates(dq.model,i)
rates(dq::DiscretisedFluidQueue) = rates(dq.model)
phases(dq::DiscretisedFluidQueue) = phases(dq.model)
N₋(dq::DiscretisedFluidQueue) = N₋(dq.model)
N₊(dq::DiscretisedFluidQueue) = N₊(dq.model)

n_intervals(dq::DiscretisedFluidQueue) = n_intervals(dq.mesh)
Δ(dq::DiscretisedFluidQueue) = Δ(dq.mesh)
Δ(dq::DiscretisedFluidQueue,k::Int) = Δ(dq.mesh,k)
"""
    n_bases_per_cell(m) = 

Return the number of basis functions on each cell of a given DiscretisedFluidQueue or Mesh.
"""
n_bases_per_cell(dq::DiscretisedFluidQueue) = n_bases_per_cell(dq.mesh)
n_bases_per_phase(dq::DiscretisedFluidQueue) = n_bases_per_phase(dq.mesh)
"""
    total_n_bases(m) 

Return the number of basis functions of a given DiscretisedFluidQueue or Mesh.
"""
total_n_bases(dq::DiscretisedFluidQueue) = n_bases_per_phase(dq)*n_phases(dq)
cell_nodes(dq::DiscretisedFluidQueue) = cell_nodes(dq.mesh)

n_bases_per_level(dq::DiscretisedFluidQueue) = n_phases(dq)*n_bases_per_cell(dq)

function qbd_idx(dq::DiscretisedFluidQueue)
    ## Make QBD index
    model = dq.model
    mesh = dq.mesh

    c = N₋(dq)
    n₊ = N₊(dq)
    n₋ = N₋(dq)
    QBDidx = zeros(Int, total_n_bases(dq) + n₊ + n₋)
    for k = 1:n_intervals(dq), i = 1:n_phases(dq), n = 1:n_bases_per_cell(dq)
        c += 1
        QBDidx[c] = (i - 1) * n_bases_per_phase(dq) + (k - 1) * n_bases_per_cell(dq) + n + n₋
    end
    QBDidx[1:n₋] = 1:n₋
    QBDidx[(end-n₊+1):end] = (total_n_bases(dq) + n₋) .+ (1:n₊)

    return QBDidx
end
