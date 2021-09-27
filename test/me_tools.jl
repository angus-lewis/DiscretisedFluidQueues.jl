@testset "ME tools" begin 
    @test MatrixExponential <: AbstractMatrixExponential
    @test_throws DimensionMismatch MatrixExponential(ones(1,2),-ones(1,1),ones(1,1))
    @test_throws DomainError MatrixExponential(ones(1,1),-[1.0 1.0],ones(1,1))
    @test_throws DomainError MatrixExponential(ones(1,1),-[1.0 1.0],ones(1,2))
    exp_rv = MatrixExponential(ones(1,1),-ones(1,1),ones(1,1))
    # pdf 
    @test DiscretisedFluidQueues._order(exp_rv)==1
    exp_pdf = pdf(exp_rv)
    @test typeof(exp_pdf(1.0))<:Float64
    r = -log.(rand(10))
    @test exp_pdf.(r)≈exp.(-r)
    @test_throws DomainError pdf([1.0 1.0],exp_rv)
    @test pdf(exp_rv,r)≈pdf(exp_rv).(r)
    @test pdf(exp(-1).*ones(1,1),exp_rv,r)≈pdf(exp_rv).(r.+1.0)
    # ccdf
    exp_ccdf = ccdf(exp_rv)
    @test typeof(exp_ccdf(1.0))<:Float64
    r = -log.(rand(10))
    @test exp_ccdf.(r)≈exp.(-r)
    @test_throws DomainError ccdf([1.0 1.0],exp_rv)
    @test ccdf(exp_rv,r)≈ccdf(exp_rv).(r)
    @test ccdf(exp(-1).*ones(1,1),exp_rv,r)≈ccdf(exp_rv).(r.+1.0)
    # cdf
    exp_cdf = cdf(exp_rv)
    @test typeof(exp_cdf(1.0))<:Float64
    r = -log.(rand(10))
    @test exp_cdf.(r)≈1.0.-exp.(-r)
    @test_throws DomainError cdf([1.0 1.0],exp_rv)
    @test cdf(exp_rv,r)≈cdf(exp_rv).(r)
    # build_me
    me_1 = build_me(cme_params[1])
    @test cdf(me_1,r)≈cdf(exp_rv,r)
    #
    me_3 = build_me(cme_params[3])
    @test typeof(me_3)<:AbstractMatrixExponential
    @test typeof(me_3)<:ConcentratedMatrixExponential
    me_3_cdf = cdf(me_3)
    @test typeof(me_3_cdf(1.0))<:Float64
    for f in fieldnames(ConcentratedMatrixExponential)
        @eval begin 
            me_3 = build_me(cme_params[3])
            @test ConcentratedMatrixExponential(3).$f==me_3.$f
        end
    end
    @test -sum(me_3.a/me_3.S)≈1.0
    # build_erlang
    erl_1 = DiscretisedFluidQueues.build_erlang(1)
    @test typeof(erl_1)<:AbstractMatrixExponential
    @test typeof(erl_1)<:MatrixExponential
    @test ccdf(erl_1,r)≈ccdf(exp_rv,r)
    erl_3 = DiscretisedFluidQueues.build_erlang(3)
    @test -sum(erl_3.a/erl_3.S)≈1.0
    e1 = zeros(1,3)
    e1[1]=1.0
    @test erl_3.a==e1
    @test erl_3.S≈[-3.0 3.0 0.0; 0.0 -3.0 3.0; 0.0 0.0 -3.0]
    @test erl_3.s≈3*e1[end:-1:1]
    # orbits 
    exp_rv_orbit_fun = DiscretisedFluidQueues.orbit(exp_rv)
    @test exp_rv_orbit_fun(2.0)≈[1.0]

    me_3_orbit_fun = DiscretisedFluidQueues.orbit(me_3)
    me_3_mean_2 = ConcentratedMatrixExponential(3; mean=2.0)
    me_3_mean_2_orbit_fun = DiscretisedFluidQueues.orbit(me_3_mean_2)
    @test me_3_mean_2_orbit_fun.(r)≈me_3_orbit_fun.(0.5*r)
    me_3_not_CME_Type = MatrixExponential(me_3.a,me_3.S,me_3.s,me_3.D)
    me_3_not_CME_Type_orbit_fun = DiscretisedFluidQueues.orbit(me_3_not_CME_Type)
    @test me_3_orbit_fun.(r)≈me_3_not_CME_Type_orbit_fun.(r)
    me_3_mean_2_not_CME_Type = 
        MatrixExponential(me_3_mean_2.a,me_3_mean_2.S,me_3_mean_2.s,me_3_mean_2.D)
    me_3_mean_2_not_CME_Type_orbit_fun = DiscretisedFluidQueues.orbit(me_3_mean_2_not_CME_Type)
    @test me_3_mean_2_not_CME_Type_orbit_fun.(r)≈me_3_not_CME_Type_orbit_fun.(0.5*r)
    # Expected orbits
    ω = abs(me_3.S[3,2])
    period = 2*pi/ω
    exp_orbit = 
        DiscretisedFluidQueues.expected_orbit_from_pdf(
            x->ccdf(me_3,x)/-sum(me_3.a/me_3.S),
            me_3,0.0,period,10000)/(1-exp(me_3.S[1,1]*period))
    stationary = (me_3.a/me_3.S)/sum(me_3.a/me_3.S)
    @test exp_orbit≈stationary atol=1e-6
    exp_orbit_from_cdf = 
        DiscretisedFluidQueues.expected_orbit_from_cdf(
            x->cdf(stationary,me_3,x)/-sum(me_3.a/me_3.S),
            me_3,0.0,period,10000)/(1-exp(me_3.S[1,1]*period))
    @test exp_orbit_from_cdf≈stationary atol=1e-6
end 