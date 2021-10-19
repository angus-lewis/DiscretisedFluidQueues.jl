using DiscretisedFluidQueues, Test
@testset "limiters" begin 
	@test DiscretisedFluidQueues.minmod(0.0,0.0,0.0)==0.0
	@test DiscretisedFluidQueues.minmod(0.0,0.0,1.0)==0.0
	@test DiscretisedFluidQueues.minmod(0.0,1.0,1.0)==0.0
	@test DiscretisedFluidQueues.minmod(0.1,1.0,1.0)==0.1
	@test DiscretisedFluidQueues.minmod(-1.0,1.0,1.0)==0.0
	@test DiscretisedFluidQueues.minmod(-1.0,-1.0,-0.1)==-0.1
	@test DiscretisedFluidQueues.minmod(-1.0,-1.0,0.0)==0.0
	@test DiscretisedFluidQueues.minmod(-1.0,1.0,0.0)==0.0
	
	V2 = DiscretisedFluidQueues.vandermonde(2)
	linear_coeffs2 = [0.0;1.0]
	delta = 1.0
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs2,V2...,delta)[1],linear_coeffs2)
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs2,V2...,delta)[2],2.0/delta^2)
	delta = 0.25
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs2,V2...,delta)[1],linear_coeffs2)
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs2,V2...,delta)[2],2.0/delta^2)
	linear_coeffs2 = [0.5;0.5]
	delta = 1.0
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs2,V2...,delta)[1],linear_coeffs2)
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs2,V2...,delta)[2],0.0)
	delta = 0.25
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs2,V2...,delta)[1],linear_coeffs2)
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs2,V2...,delta)[2],0.0)
		
	V3 = DiscretisedFluidQueues.vandermonde(3)
	linear_coeffs3 = [1/6; 2/3; 1/6]
	delta = 1.0
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs3,V3...,delta)[1],linear_coeffs3)
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs3,V3...,delta)[2],0.0)
	delta = 0.25
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs3,V3...,delta)[1],linear_coeffs3)
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs3,V3...,delta)[2],0.0)
		
	linear_coeffs3 = [0.0; 2/3; 1/3]
	delta = 1.0
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs3,V3...,delta)[1],linear_coeffs3)
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs3,V3...,delta)[2],2.0/delta^2)
	delta = 0.25
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs3,V3...,delta)[1],linear_coeffs3)
	@test isapprox(DiscretisedFluidQueues.linear(linear_coeffs3,V3...,delta)[2],2.0/delta^2)
	
	quadratic_coeffs3 = [0.0; 1.0; 0.0]
	linear_coeffs3 = [1/6; 2/3; 1/6]
	delta = 1.0
	@test isapprox(DiscretisedFluidQueues.linear(quadratic_coeffs3,V3...,delta)[1],linear_coeffs3)
	@test isapprox(DiscretisedFluidQueues.linear(quadratic_coeffs3,V3...,delta)[2],0.0)
	delta = 0.25
	@test isapprox(DiscretisedFluidQueues.linear(quadratic_coeffs3,V3...,delta)[1],linear_coeffs3)
	@test isapprox(DiscretisedFluidQueues.linear(quadratic_coeffs3,V3...,delta)[2],0.0)
	
	T_limiter = [0.0][:,:]
	c_limiter = [0.0]
	m_limiter = FluidQueue(T_limiter,c_limiter)
	nodes = collect(-1.0:0.1:1.0)
	mesh_limiter = DGMesh(nodes,3)
	dq = DiscretisedFluidQueue(m_limiter,mesh_limiter)
	d0 = SFMDistribution((x,i)->x,dq)
	# limiting this linear function should do nothing
	@test isapprox(limit(d0).coeffs,d0.coeffs; atol=1e-8)
	@test isapprox(limit(d0).coeffs[:],
		GeneralisedMUSCL.fun(d0.coeffs[:],GeneralisedMUSCL.generate_params(d0.dq)...);
		atol = 1e-8)
	@test d0.coeffs[:]==NoLimiter.fun(d0.coeffs[:],NoLimiter.generate_params(d0.dq)...)
	
	mesh_limiter = DGMesh(nodes,4)
	dq = DiscretisedFluidQueue(m_limiter,mesh_limiter)
	d0 = interior_point_mass(0.05,1,dq)
	# limiting this cubic approx to a point mass should do something
	@test !isapprox(limit(d0).coeffs,d0.coeffs; atol=1e-8)
	@test d0.coeffs[:]==NoLimiter.fun(d0.coeffs[:],NoLimiter.generate_params(d0.dq)...)
end