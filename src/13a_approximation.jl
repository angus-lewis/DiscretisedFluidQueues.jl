
function SFMDistribution(pdf::Function,model::Model,mesh::DGMesh)
    cellnodes = cell_nodes(mesh)
    n₋ = N₋(model.S)
    coeffs = zeros(n_bases(mesh),n_intervals(mesh),n_phases(model))
    for i in phases(model)
        for cell in 1:n_intervals(mesh)
            nodes = cellnodes[:,cell]
            weights = gauss_lobatto_weights(nodes[1],nodes[end],length(nodes))
            coeffs[:,cell,i] = pdf.(nodes,i).*weights
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(model.S))]
    return SFMDistribution(coeffs,model,mesh)
end

function interior_point_mass(x::Float64,i::Int,model::Model,mesh::DGMesh)
    (x<=mesh.nodes[1])&&throw(DomainError("x is not in interior"))
    (x>=mesh.nodes[end])&&throw(DomainError("x is not in interior"))
    if _has_right_boundary(model.S,i) 
        cell_idx, cellnodes, coeff_idx = _get_coeff_index_pos(x,i,mesh,model) 
    elseif _has_left_boundary(model.S,i)
        cell_idx, cellnodes, coeff_idx = _get_coeff_index_neg(x,i,mesh,model) 
    end
    n₊ = N₊(model.S)
    n₋ = N₋(model.S)
    coeffs = zeros(1,n₊+n₋+total_n_bases(mesh)*n_phases(model))
    nodes = cell_nodes(mesh)[:,cell_idx]
    coeffs[coeff_idx] = lagrange_polynomials(nodes,x)
    return SFMDistribution(coeffs,model,mesh)
end

function left_point_mass(i::Int,model::Model,mesh::Mesh)
    _has_right_boundary(model.S,i)&&throw(DomainError("only phases with lpm=true have left point masses"))
    n₊ = N₊(model.S)
    n₋ = N₊(model.S)
    coeffs = zeros(1,n₊+n₋+total_n_bases(mesh)*n_phases(model))
    nᵢ = N₋(model.S[1:i])
    coeffs[nᵢ] = 1.0
    return SFMDistribution(coeffs,model,mesh)
end

function right_point_mass(i::Int,model::Model,mesh::Mesh)
    _has_left_boundary(model.S,i)&&throw(DomainError("only phases with rpm=false have left point masses"))
    n₊ = N₊(model.S)
    n₋ = N₊(model.S)
    coeffs = zeros(1,n₊+n₋+total_n_bases(mesh)*n_phases(model))
    nᵢ = N₊(model.S[1:i])
    coeffs[end-n₊+nᵢ] = 1.0
    return SFMDistribution(coeffs,model,mesh)
end

function SFMDistribution(pdf::Function,model::Model,mesh::FVMesh, fun_evals::Int=6)
    n₋ = N₋(model.S)
    coeffs = zeros(n_bases(mesh),n_intervals(mesh),n_phases(model))
    for i in phases(model)
        for cell in 1:n_intervals(mesh)
            a,b = mesh.nodes[cell:cell+1]
            quad = gauss_lobatto_quadrature(x->pdf(x,i),a,b,fun_evals)
            coeffs[1,cell,i] = quad./Δ(mesh,cell)
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(model.S))]
    return SFMDistribution(coeffs,model,mesh)
end

function interior_point_mass(x::Float64,i::Int,model::Model,mesh::FVMesh)
    if _has_right_boundary(model.S,i)
        cell_idx, ~, ~ = _get_coeff_index_pos(x,i,mesh,model) 
    elseif _has_left_boundary(model.S,i)
        cell_idx, ~, ~ = _get_coeff_index_neg(x,i,mesh,model) 
    end
    n₊ = N₊(model.S)
    n₋ = N₋(model.S)
    coeffs = zeros(1,n₊+n₋+total_n_bases(mesh)*n_phases(model))
    coeffs[cell_idx] = 1.0./Δ(mesh,cell_idx)
    return SFMDistribution(coeffs,model,mesh)
end

## constructors for FRAPMesh
function SFMDistribution_from_cdf(cdf::Function,model::Model,mesh::FRAPMesh; fun_evals::Int=10)
    n₋ = N₋(model.S)
    coeffs = zeros(n_bases(mesh),n_intervals(mesh),n_phases(model))
    for i in phases(model)
        for cell in 1:n_intervals(mesh)
            a,b = mesh.nodes[cell:cell+1]
            if _has_right_boundary(model.S,i)
                o = expected_orbit_from_cdf(x->cdf(Δ(mesh,cell)-x,i),a,b,fun_evals) 
            elseif _has_left_boundary(model.S,i)
                o = expected_orbit_from_cdf(x->cdf(x,i),a,b,fun_evals) 
            end
            coeffs[:,cell,i] = o
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(model.S))]
    return SFMDistribution(coeffs,model,mesh)
end

function SFMDistribution(pdf::Function,model::Model,mesh::FRAPMesh; fun_evals=100)
    n₋ = N₋(model.S)
    coeffs = zeros(n_bases(mesh),n_intervals(mesh),n_phases(model))
    for i in phases(model)
        for cell in 1:n_intervals(mesh)
            a,b = mesh.nodes[cell:cell+1]
            if _has_right_boundary(model.S,i)
                o = expected_orbit_from_pdf(x->(b-a)*pdf(a+x*(b-a),i),mesh.me,0.0,1.0,fun_evals) 
            else
                o = expected_orbit_from_pdf(x->(b-a)*pdf(b-x*(b-a),i),mesh.me,0.0,1.0,fun_evals) 
            end
            coeffs[:,cell,i] = o
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(model.S))]
    return SFMDistribution(coeffs,model,mesh)
end

function interior_point_mass(x::Float64,i::Int,model::Model,mesh::FRAPMesh)
    if _has_right_boundary(model.S,i) 
        cell_idx, ~, coeff_idx = _get_coeff_index_pos(x,i,mesh,model) 
        yₖ = mesh.nodes[cell_idx]
        d = x-yₖ
    elseif _has_left_boundary(model.S,i)
        cell_idx, ~, coeff_idx = _get_coeff_index_neg(x,i,mesh,model)
        yₖ₊₁ = mesh.nodes[cell_idx+1]
        d = yₖ₊₁-x
    end
    o = orbit(mesh.me,d/Δ(mesh,cell_idx))
    coeffs = zeros(1,N₊(model.S)+N₋(model.S)+total_n_bases(mesh)*n_phases(model))
    coeffs[coeff_idx] = o
    return SFMDistribution(coeffs,model,mesh)
end