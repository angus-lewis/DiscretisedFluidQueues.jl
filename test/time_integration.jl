@testset "time integration" begin
    D = [-1.0 5.0; -5.0 -1.0]
    x0 = [1.0 0.0]
    test_data = x0*exp(D)
    e = Euler(1e-4)
    rk4 = RungeKutta4(1e-4)
    @test test_data≈integrate_time(x0,D,1.0,e) atol=e.step_size*10.0
    @test test_data≈integrate_time(x0,D,1.0,rk4) atol=rk4.step_size*10.0
end