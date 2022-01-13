@testset "sim" begin
    import StableRNGs
    rng = StableRNGs.StableRNG(1)
    stopping_time = fixed_time(3.2)
    n_sims = 100_000
    initial_condition = (φ=ones(Int,n_sims),X=zeros(n_sims))
    sims = simulate(model,stopping_time,initial_condition,rng)
    f(x,i) = cdf(sims)(x,i)
    @test sum(f.(10.0,phases(model)))≈1.0 
    p_3_2 = ([1.0 0 0] * exp(model.T*3))[:]
    @test f.(10.0,phases(model))≈p_3_2 rtol=1e-3
end
