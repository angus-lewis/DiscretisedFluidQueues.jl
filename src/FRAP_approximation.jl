"""
    FRAPMesh{T} <: Mesh{T}

A QBD-RAP discretisation scheme for a DiscretisedFluidQueue. 

# Arguments
- `nodes`: the cell edges
- `me`: the MatrixExponential used to approximate model the fluid queue on each cell.
"""
struct FRAPMesh{T} <: Mesh{T}
    nodes::T
    me::AbstractMatrixExponential
    function FRAPMesh{T}(nodes::T,me::AbstractMatrixExponential) where T
        (nodes[1]!=0.0)&&throw(DomainError("first node must be 0"))
        return new{T}(nodes,me)
    end
end 
FRAPMesh(nodes::T,me::AbstractMatrixExponential) where T = 
    FRAPMesh{T}(nodes::T,me::AbstractMatrixExponential)
FRAPMesh{T}(nodes::T,n_bases::Int) where T = FRAPMesh{T}(nodes,build_me(cme_params[n_bases]))
FRAPMesh(nodes::T,n_bases::Int) where T = FRAPMesh{T}(nodes,build_me(cme_params[n_bases]))

"""

    n_bases_per_cell(mesh::FRAPMesh)
    
Number of bases in a cell
"""
n_bases_per_cell(mesh::FRAPMesh) = _order(mesh.me)


"""

    cell_nodes(mesh::FRAPMesh)

The cell centre
"""
cell_nodes(mesh::FRAPMesh) = Array(((mesh.nodes[1:end-1] + mesh.nodes[2:end]) / 2 )')

function build_lazy_generator(
    dq::DiscretisedFluidQueue{FRAPMesh{T}};
    v::Bool=false,
) where T
    me = dq.mesh.me
    blocks = (me.s*me.a, me.S, me.s*me.a)
    boundary_flux = OneBoundaryFlux(me.s[:],me.a[:])
    D = me.D
    out = LazyGenerator(dq,blocks,boundary_flux,D)
    v && println("UPDATE: LazyGenerator object created with keys ", keys(out))
    return out
end
