
function SFMDistribution(pdf::Function,dq::DiscretisedFluidQueue{DGMesh})
    cellnodes = cell_nodes(dq)
    n₋ = N₋(dq)
    coeffs = zeros(n_bases_per_cell(dq),n_intervals(dq),n_phases(dq))
    for i in phases(dq)
        for cell in 1:n_intervals(dq)
            nodes = cellnodes[:,cell]
            weights = gauss_lobatto_weights(nodes[1],nodes[end],length(nodes))
            coeffs[:,cell,i] = pdf.(nodes,i).*weights
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(dq))]
    return SFMDistribution(coeffs,dq)
end

function interior_point_mass(x::Float64,i::Int,dq::DiscretisedFluidQueue{DGMesh})
    (x<=dq.mesh.nodes[1])&&throw(DomainError("x is not in interior"))
    (x>=dq.mesh.nodes[end])&&throw(DomainError("x is not in interior"))
    if _has_right_boundary(dq.model.S,i) 
        cell_idx, cellnodes, coeff_idx = _get_coeff_index_pos(x,i,dq) 
    elseif _has_left_boundary(dq.model.S,i)
        cell_idx, cellnodes, coeff_idx = _get_coeff_index_neg(x,i,dq) 
    end
    n₊ = N₊(dq)
    n₋ = N₋(dq)
    coeffs = zeros(1,n₊+n₋+total_n_bases(dq))
    nodes = cell_nodes(dq.mesh)[:,cell_idx]
    coeffs[coeff_idx] = lagrange_polynomials(nodes,x)
    return SFMDistribution(coeffs,dq)
end

function left_point_mass(i::Int,dq::DiscretisedFluidQueue)
    _has_right_boundary(dq.model.S,i)&&throw(DomainError("only phases with lpm=true have left point masses"))
    n₊ = N₊(dq)
    n₋ = N₊(dq)
    coeffs = zeros(1,n₊+n₋+total_n_bases(dq))
    nᵢ = N₋(dq.model.S[1:i])
    coeffs[nᵢ] = 1.0
    return SFMDistribution(coeffs,dq)
end

function right_point_mass(i::Int,dq::DiscretisedFluidQueue)
    _has_left_boundary(dq.model.S,i)&&throw(DomainError("only phases with rpm=false have left point masses"))
    n₊ = N₊(dq)
    n₋ = N₊(dq)
    coeffs = zeros(1,n₊+n₋+total_n_bases(dq))
    nᵢ = N₊(dq.model.S[1:i])
    coeffs[end-n₊+nᵢ] = 1.0
    return SFMDistribution(coeffs,dq)
end

function SFMDistribution(pdf::Function,dq::DiscretisedFluidQueue{FVMesh}, fun_evals::Int=6)
    n₋ = N₋(dq)
    coeffs = zeros(n_bases_per_cell(dq),n_intervals(dq),n_phases(dq))
    for i in phases(dq)
        for cell in 1:n_intervals(dq)
            a,b = dq.mesh.nodes[cell:cell+1]
            quad = gauss_lobatto_quadrature(x->pdf(x,i),a,b,fun_evals)
            coeffs[1,cell,i] = quad./Δ(dq,cell)
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(dq))]
    return SFMDistribution(coeffs,dq)
end

function interior_point_mass(x::Float64,i::Int,dq::DiscretisedFluidQueue{FVMesh})
    if _has_right_boundary(dq.model.S,i)
        cell_idx, ~, ~ = _get_coeff_index_pos(x,i,dq) 
    elseif _has_left_boundary(dq.model.S,i)
        cell_idx, ~, ~ = _get_coeff_index_neg(x,i,dq) 
    end
    n₊ = N₊(dq)
    n₋ = N₋(dq)
    coeffs = zeros(1,n₊+n₋+total_n_bases(dq))
    coeffs[cell_idx] = 1.0./Δ(dq,cell_idx)
    return SFMDistribution(coeffs,dq)
end

## constructors for FRAPMesh
function SFMDistribution_from_cdf(cdf::Function,dq::DiscretisedFluidQueue{FRAPMesh}; fun_evals::Int=10)
    n₋ = N₋(dq)
    coeffs = zeros(n_bases_per_cell(dq),n_intervals(dq),n_phases(dq))
    for i in phases(dq)
        for cell in 1:n_intervals(dq)
            a,b = dq.mesh.nodes[cell:cell+1]
            if _has_right_boundary(dq.model.S,i)
                o = expected_orbit_from_cdf(x->cdf(Δ(dq,cell)-x,i),a,b,fun_evals) 
            elseif _has_left_boundary(dq.model.S,i)
                o = expected_orbit_from_cdf(x->cdf(x,i),a,b,fun_evals) 
            end
            coeffs[:,cell,i] = o
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(dq.model.S))]
    return SFMDistribution(coeffs,dq)
end

function SFMDistribution(pdf::Function,dq::DiscretisedFluidQueue{FRAPMesh}; fun_evals=100)
    n₋ = N₋(dq)
    coeffs = zeros(n_bases_per_cell(dq),n_intervals(dq),n_phases(dq))
    for i in phases(dq)
        for cell in 1:n_intervals(dq)
            a,b = dq.mesh.nodes[cell:cell+1]
            if _has_right_boundary(dq.model.S,i)
                o = expected_orbit_from_pdf(x->(b-a)*pdf(a+x*(b-a),i),dq.mesh.me,0.0,1.0,fun_evals) 
            else
                o = expected_orbit_from_pdf(x->(b-a)*pdf(b-x*(b-a),i),dq.mesh.me,0.0,1.0,fun_evals) 
            end
            coeffs[:,cell,i] = o
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(dq))]
    return SFMDistribution(coeffs,dq)
end

function interior_point_mass(x::Float64,i::Int,dq::DiscretisedFluidQueue{FRAPMesh})
    if _has_right_boundary(dq.model.S,i) 
        cell_idx, ~, coeff_idx = _get_coeff_index_pos(x,i,dq) 
        yₖ = dq.mesh.nodes[cell_idx]
        d = x-yₖ
    elseif _has_left_boundary(dq.model.S,i)
        cell_idx, ~, coeff_idx = _get_coeff_index_neg(x,i,dq)
        yₖ₊₁ = dq.mesh.nodes[cell_idx+1]
        d = yₖ₊₁-x
    end
    o = orbit(dq.mesh.me,d/Δ(dq.mesh,cell_idx))
    coeffs = zeros(1,N₊(dq)+N₋(dq)+total_n_bases(dq))
    coeffs[coeff_idx] = o
    return SFMDistribution(coeffs,dq)
end