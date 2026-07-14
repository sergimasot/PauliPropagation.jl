using Random
using Test

@testset "VectorPauliSum sortedprefix survives truncate! correctly" begin
    # Truncating can remove terms from inside an already-sorted head while later, unsorted terms
    # survive untouched. The new sorted-prefix count must match how many head terms actually
    # survived -- not just clamped to the old count, and not reset to 0.
    nq = 6
    Random.seed!(11)

    all_pstrs = PauliString[]
    while length(all_pstrs) < 7
        p = PauliString(nq, rand([:X, :Y, :Z], nq), 1:nq, 1.0)
        if p.term ∉ getfield.(all_pstrs, :term)
            push!(all_pstrs, p)
        end
    end
    all_terms = getfield.(all_pstrs, :term)

    n_head, n_tail = 5, 2
    head_terms = sort(all_terms[1:n_head])
    tail_terms = all_terms[n_head+1:n_head+n_tail]

    # a mid-head term is below threshold; everything else, including both tail terms, is above it
    head_coeffs = [1.0, 1.0, 1e-8, 1.0, 1.0]
    tail_coeffs = [1.0, 1.0]

    vpsum = VectorPauliSum(nq, vcat(head_terms, tail_terms), vcat(head_coeffs, tail_coeffs), n_head)
    prop_cache = VectorPauliPropagationCache(vpsum)

    truncate!(prop_cache; min_abs_coeff=1e-6)

    @test PauliPropagation.activesize(prop_cache) == n_head + n_tail - 1
    @test PauliPropagation.sortedprefix(mainsum(prop_cache)) == n_head - 1

    surviving_terms = vcat(head_terms[1:2], head_terms[4:5], tail_terms)
    for trm in surviving_terms
        @test PauliPropagation.getmergedcoeff(mainsum(prop_cache), trm) == 1.0
    end
    @test PauliPropagation.getmergedcoeff(mainsum(prop_cache), head_terms[3]) == 0.0
end

@testset "VectorPauliSum matches PauliSum: mixed CliffordGate + PauliRotation + truncation" begin
    # A CliffordGate leaves the terms unsorted without merging them, so the next PauliRotation can
    # produce duplicates anywhere in that unsorted region, not just in its own new tail. A small,
    # simple circuit can pass by luck without ever hitting this; efficientsu2circuit's mix of
    # single-qubit rotations and CNOT blocks reliably does.
    nq = 6
    topology = rectangletopology(2, 3; periodic=true)

    for nl in (1, 3, 5)
        circuit = efficientsu2circuit(nq, nl; topology=topology)
        Random.seed!(20 + nl)
        thetas = randn(countparameters(circuit))

        for max_weight in (0, 1, 3, 6)
            pstr = PauliString(nq, rand([:I, :X, :Y, :Z]), rand(1:nq))
            dnum = propagate(circuit, pstr, thetas; min_abs_coeff=0, max_weight=max_weight)
            dvec = propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff=0, max_weight=max_weight)

            @test dnum ≈ dvec
            @test overlapwithzero(dnum) ≈ overlapwithzero(dvec)
        end
    end
end

@testset "Sort-tail-merge specialization: PauliRotation" begin
    nq = 6
    nl = 5
    topo = bricklayertopology(nq; periodic=false)
    circuit = hardwareefficientcircuit(nq, nl; topology=topo)

    Random.seed!(3)
    thetas = randn(countparameters(circuit))

    pstr = PauliString(nq, :Z, 3)
    min_abs_coeff = 1e-4

    dnum = propagate(circuit, pstr, thetas; min_abs_coeff=min_abs_coeff)
    dvec = propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff=min_abs_coeff)

    @test dvec isa VectorPauliSum
    @test dnum ≈ dvec
    @test overlapwithzero(dnum) ≈ overlapwithzero(dvec)
end

@testset "Sort-tail-merge specialization: ImaginaryPauliRotation" begin
    nq = 5
    circuit = [ImaginaryPauliRotation(rand([:X, :Y, :Z], nq), 1:nq) for _ in 1:20]

    Random.seed!(4)
    taus = 0.1 * rand(countparameters(circuit))

    pstr = PauliString(nq, :I, 1)  # identity-supported start keeps the identity coefficient well-defined

    for normalize_coeffs in (true, false)
        dnum = propagate(circuit, pstr, taus; heisenberg=false, normalize_coeffs=normalize_coeffs, min_abs_coeff=1e-8)
        dvec = propagate(circuit, VectorPauliSum(pstr), taus; heisenberg=false, normalize_coeffs=normalize_coeffs, min_abs_coeff=1e-8)

        @test dvec isa VectorPauliSum
        @test dnum ≈ dvec
        @test overlapwithzero(dnum) ≈ overlapwithzero(dvec)
    end
end

@testset "thread=false matches thread=true" begin
    nq = 6
    nl = 5
    topo = bricklayertopology(nq; periodic=false)
    circuit = [
        CliffordGate(:H, 1),
        hardwareefficientcircuit(nq, nl; topology=topo)...,
        CliffordGate(:CNOT, [2, 3]),
    ]

    Random.seed!(5)
    thetas = randn(countparameters(circuit))
    pstr = PauliString(nq, :Z, 3)
    min_abs_coeff = 1e-4

    d_thread = propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff=min_abs_coeff, thread=true)
    d_nothread = propagate(circuit, VectorPauliSum(pstr), thetas; min_abs_coeff=min_abs_coeff, thread=false)

    @test d_thread == d_nothread
    @test overlapwithzero(d_thread) == overlapwithzero(d_nothread)
end

@testset "Sort-tail-merge specialization: multi-task merge path" begin
    # A sorted head large enough that AK.TaskPartitioner splits the merge into more than one
    # task (only reachable with Threads.nthreads() > 1 and a few thousand sorted elements).
    # The multi-task write path assigns write offsets via a global prefix sum while sorted-tail
    # storage sits at fixed positions, so chunk boundaries have to line up exactly; tail terms
    # are drawn so that some collide with head terms and some repeat within the tail itself,
    # exercising both head/tail and intra-tail merges that straddle those boundaries.
    nq = 20
    TT = PauliPropagation.getinttype(nq)
    Random.seed!(7)

    n_head = 6000
    n_tail = 3000

    head_terms = TT[]
    head_set = Set{TT}()
    while length(head_set) < n_head
        t = rand(TT(0):TT(4^nq - 1))
        if t ∉ head_set
            push!(head_set, t)
            push!(head_terms, t)
        end
    end
    sort!(head_terms)
    head_coeffs = rand(n_head)

    tail_terms = TT[]
    tail_coeffs = Float64[]
    for _ in 1:n_tail
        t = rand() < 0.3 ? rand(head_terms) : rand(TT(0):TT(4^nq - 1))
        push!(tail_terms, t)
        push!(tail_coeffs, rand())
    end

    reference = Dict{TT,Float64}()
    for (t, c) in zip(vcat(head_terms, tail_terms), vcat(head_coeffs, tail_coeffs))
        reference[t] = get(reference, t, 0.0) + c
    end

    vpsum = VectorPauliSum(nq, vcat(head_terms, tail_terms), vcat(head_coeffs, tail_coeffs), n_head)
    prop_cache = VectorPauliPropagationCache(vpsum)

    PB = PauliPropagation.PropagationBase
    task_partitioner = PB.AK.TaskPartitioner(n_head, PB.maxtasks(true), PB._TAILMERGE_MIN_ELEMS_PER_TASK)
    if Threads.nthreads() > 1
        @test task_partitioner.num_tasks > 1
    end

    merge!(prop_cache)

    result_terms = PauliPropagation.activeterms(prop_cache)
    result_coeffs = PauliPropagation.activecoeffs(prop_cache)

    @test length(result_terms) == length(reference)
    @test issorted(result_terms)
    @test PauliPropagation.sortedprefix(mainsum(prop_cache)) == length(result_terms)
    @test Set(result_terms) == Set(keys(reference))
    @test all(isapprox(c, reference[t]) for (t, c) in zip(result_terms, result_coeffs))
end

@testset "thread=false is safe to nest inside external threading" begin
    # thread=false must not spawn its own tasks, so running several propagations concurrently
    # inside an outer Threads.@threads loop should give exactly the same results as running them
    # one at a time.
    nq = 6
    nl = 5
    topo = bricklayertopology(nq; periodic=false)
    circuit = hardwareefficientcircuit(nq, nl; topology=topo)
    min_abs_coeff = 1e-4

    n_runs = 2 * Threads.nthreads()
    all_thetas = [randn(MersenneTwister(100 + i), countparameters(circuit)) for i in 1:n_runs]
    all_pstrs = [PauliString(nq, rand(MersenneTwister(200 + i), [:X, :Y, :Z]), rand(MersenneTwister(300 + i), 1:nq)) for i in 1:n_runs]

    reference = [propagate(circuit, all_pstrs[i], all_thetas[i]; min_abs_coeff=min_abs_coeff, thread=false) for i in 1:n_runs]

    results = Vector{Any}(undef, n_runs)
    Threads.@threads for i in 1:n_runs
        results[i] = propagate(circuit, all_pstrs[i], all_thetas[i]; min_abs_coeff=min_abs_coeff, thread=false)
    end

    for i in 1:n_runs
        @test results[i] == reference[i]
    end
end
