@testset "time integration" begin
    D = [-1.0 5.0; -5.0 -1.0]
    x0 = [1.0 0.0]
    test_data = x0*exp(D)
    e = ForwardEuler(1e-4)
    heuns = Heuns(1e-4)
    ssprk3 = StableRK3(1e-4)
    ssprk4 = StableRK4(1e-4)
    @test test_data≈integrate_time(x0,D,1.0,e) atol=e.step_size*10.0
    @test test_data≈integrate_time(x0,D,1.0,heuns) atol=heuns.step_size*10.0
    @test test_data≈integrate_time(x0,D,1.0,ssprk3) atol=ssprk3.step_size*10.0
    @test test_data≈integrate_time(x0,D,1.0,ssprk3) atol=ssprk4.step_size*10.0
end
