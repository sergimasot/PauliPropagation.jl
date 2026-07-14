using Random
using Test
using LinearAlgebra

using PauliPropagation.Performance

@testset "Performance module is opt-in and does not change stock propagate" begin
    # `propagate`/`propagate!` with `fused=false` should exactly replicate main library behavior. 
    nq = 6
    nl = 3
    topo = bricklayertopology(nq; periodic=false)
    circuit = hardwareefficientcircuit(nq, nl; topology=topo)

    Random.seed!(1)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 3)
    min_abs_coeff = 1e-4

    stock_dict = propagate(circuit, pstr, thetas; min_abs_coeff)
    stock_vec = propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff)

    @test Performance.propagate(circuit, pstr, thetas; min_abs_coeff, fused=false) == stock_dict
    @test Performance.propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff, fused=false) == stock_vec

    # calling propagate with fused=true elsewhere must not retroactively change what a plain,
    # fused-less propagate call returns
    Performance.propagate(circuit, pstr, thetas; min_abs_coeff, fused=true)
    @test propagate(circuit, pstr, thetas; min_abs_coeff) == stock_dict
end

@testset "fused Dict, fused Vector and stock propagation agree exactly without coefficient truncation" begin
    # truncation by max_weight should not affect any results during propagation
    nq = 6
    topo = rectangletopology(2, 3; periodic=true)

    for nl in (1, 3, 5)
        circuit = efficientsu2circuit(nq, nl; topology=topo)
        Random.seed!(10 + nl)
        thetas = randn(countparameters(circuit))
        pstr = PauliString(nq, rand([:X, :Y, :Z]), rand(1:nq))

        for max_weight in (2, 4, nq)
            stock = propagate(circuit, pstr, thetas; min_abs_coeff=0.0, max_weight)
            dict_fused = Performance.propagate(circuit, pstr, thetas; min_abs_coeff=0.0, max_weight, fused=true)
            vec_fused_sum = Performance.propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff=0.0, max_weight, fused=true)
            vec_fused = PauliSum(nq, Dict(zip(paulis(vec_fused_sum), coefficients(vec_fused_sum))))

            @test dict_fused == stock
            @test vec_fused == stock
        end
    end
end

@testset "fused Dict and fused Vector agree with each other and with stock within a small tolerance under coefficient truncation" begin
    # min_abs_coeff makes propagation results differ, but it should not be by much.
    nq = 8
    topo = bricklayertopology(nq; periodic=false)
    tol = 3e-2

    for nl in (3, 5)
        circuit = hardwareefficientcircuit(nq, nl; topology=topo)

        Random.seed!(42)
        thetas = randn(countparameters(circuit))
        pstr = PauliString(nq, :Z, 3)
        min_abs_coeff = 1e-4

        stock = propagate(circuit, pstr, thetas; min_abs_coeff)
        dict_fused = Performance.propagate(circuit, pstr, thetas; min_abs_coeff, fused=true)
        vec_fused_sum = Performance.propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff, fused=true)
        vec_fused = PauliSum(nq, Dict(zip(paulis(vec_fused_sum), coefficients(vec_fused_sum))))

        relnorm(a, b) = norm(a - b) / norm(a)

        @test relnorm(stock, dict_fused) < tol
        @test relnorm(stock, vec_fused) < tol
        @test relnorm(dict_fused, vec_fused) < tol

        @test isapprox(overlapwithzero(stock), overlapwithzero(dict_fused); atol=tol)
        @test isapprox(overlapwithzero(stock), overlapwithzero(vec_fused); atol=tol)
    end
end

@testset "fused Dict PauliNoise matches stock exactly" begin
    # PauliNoise should not be affected by the fuse and Dict internals usage
    nq = 6

    Random.seed!(2)
    pstr = PauliString(nq, rand([:X, :Y, :Z], nq), 1:nq)
    noise_circuit = [DepolarizingNoise(qind, 0.03 + 0.01 * qind) for qind in 1:nq]
    min_abs_coeff = 1e-8

    stock = propagate(noise_circuit, pstr; min_abs_coeff)
    dict_fused = Performance.propagate(noise_circuit, pstr; min_abs_coeff, fused=true)

    @test dict_fused == stock
end

@testset "fused Vector: thread=false matches thread=true on a multi-task-sized propagation" begin
    # AK.TaskPartitioner only splits into multiple tasks once the active range is large enough, so
    # a small circuit would silently skip the multi-task code path. This one grows past 10^5 terms.
    nq = 14
    topo = bricklayertopology(nq; periodic=false)
    circuit = hardwareefficientcircuit(nq, 6; topology=topo)

    Random.seed!(9)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 3)
    min_abs_coeff = 1e-6

    d_thread = Performance.propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff, fused=true, thread=true)
    d_nothread = Performance.propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff, fused=true, thread=false)

    @test length(d_thread) > 1024  # sanity check that this circuit actually exercises multiple tasks
    @test d_thread == d_nothread
    @test overlapwithzero(d_thread) == overlapwithzero(d_nothread)
end
