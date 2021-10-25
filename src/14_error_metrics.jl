kolmogorov_smirnov(F1::Function,F2::Function,grid) = 
    maximum(abs.(F1.(grid)-F2.(grid)))

function kolmogorov_smirnov(d::SFMDistribution,F2::Function)
    grid = range(d.dq.mesh.nodes[1],d.dq.mesh.nodes[end],length=2001)
    F1 = cdf(d)
    M = 0.0
    for i in 1:n_phases(d.dq)
        m = maximum(abs.(F1.(grid)-F2.(grid)))
        (m>M)&&(M=m) 
    end
    return M
end
kolmogorov_smirnov(d::SFMDistribution,s::Simulation) = kolmogorov_smirnov(d,cdf(s))
kolmogorov_smirnov(d::SFMDistribution,s::SFMDistribution) = kolmogorov_smirnov(d,cdf(s))

function Lp_cell_probs(d::SFMDistribution,F2::Function, p=1)
    grid = (d.mesh.nodes[1:end-1]+d.mesh.nodes[2:end])./2.0
    F1 = cell_probs(d)
    M = 0.0
    for i in 1:n_phases(d.dq)
        m = sum(abs.(F1.(grid,i)-F2(grid,i)).^p)
        M += m
    end
    return M
end
Lp_cell_probs(d::SFMDistribution,s::Simulation,p=1) = Lp_cell_probs(d,cell_probs(s),p)
Lp_cell_probs(d::SFMDistribution,s::SFMDistribution,p=1) = Lp_cell_probs(d,cell_probs(s),p)

function Lp_pdf(d::SFMDistribution,f2::Function, p=1)
    grid = range(d.dq.mesh.nodes[1],d.dq.mesh.nodes[end],length=2001)
    f1 = pdf(d)
    M = 0.0
    for i in 1:n_phases(d.dq)
        fn_evals = abs.(f1.(grid,i)-f2(grid,i)).^p
        m = sum((fn_evals[1:end-1]+fn_evals[2:end])./2.0 .* diff(grid))
        M += m
    end
    return M
end
Lp_pdf(d::SFMDistribution,s::SFMDistribution,p=1) = Lp_pdf(d,pdf(s),p)
