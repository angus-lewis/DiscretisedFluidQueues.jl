@testset "bounded queues" begin 
    mesh = @eval $i
    P_lwr = zeros(sum(rates(model.S).<0.0),n_phases(model))
    P_lwr[end] = 1.0
    P_upr = zeros(sum(rates(model.S).>0.0),n_phases(model))
    P_upr[2] = 1.0
    model_bnd = BoundedFluidQueue(T,S,P_lwr,P_upr)

    dq = @eval DiscretisedFluidQueue(model,mesh)
    dq_bnd = @eval DiscretisedFluidQueue(model_bnd,mesh)
    B = build_lazy_generator(dq)
    B_bnd = build_lazy_generator(dq_bnd)
    @test B==B_bnd
    @test B_bnd*B_bnd≈B*B
    @test Matrix{Float64}(I(size(B_bnd,1)))*B_bnd≈Matrix{Float64}(I(size(B_bnd,1)))*B
end 