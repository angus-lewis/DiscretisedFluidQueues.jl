"""
    DiscretisedFluidQueue{T <: Mesh}

A container to hold a FluidQueue and a discretisation scheme (a mesh). 
Parametric type depending on the type of discretisation scheme (<:Mesh)
# Arguments:
- `model::Model`: a FluidQueue
- `mesh::T`: either a DGMesh, FVMesh or FRAPMesh specifying the discretisation scheme for the 
    continuous level of the fluid queue, X(t). 
"""
struct DiscretisedFluidQueue{T<:Mesh}
    model::Model
    mesh::T
end

n_phases(dq::DiscretisedFluidQueue) = n_phases(dq.model)
rates(dq::DiscretisedFluidQueue,i::Int) = rates(dq.model,i)
rates(dq::DiscretisedFluidQueue) = rates(dq.model)
phases(dq::DiscretisedFluidQueue) = phases(dq.model)
N₋(dq::DiscretisedFluidQueue) = N₋(dq.model)
N₊(dq::DiscretisedFluidQueue) = N₊(dq.model)

# _has_left_boundary(i::Phase) = i.lpm
# _has_right_boundary(i::Phase) = i.rpm
# _has_left_boundary(S::PhaseSet,i::Int) = S[i].lpm
# _has_right_boundary(S::PhaseSet,i::Int) = S[i].rpm
# _has_left_boundary(S::PhaseSet) = _has_left_boundary.(S)
# _has_right_boundary(S::PhaseSet) = _has_right_boundary.(S)
# N₋(S::PhaseSet) = sum(_has_left_boundary(S))
# N₊(S::PhaseSet) = sum(_has_right_boundary(S))

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
