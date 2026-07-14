###
##
# Variants of `applymergetruncate!` for VectorPauliSum that truncate during gate application.
# This may yield slightly different results compared to normal functionality.
# This will only function for CPU, currently.
##
###

@inline function _fusedtruncfunc(pstr, coeff; min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc)
    PauliPropagation.truncatemincoeff(coeff, min_abs_coeff) && return true
    PauliPropagation.truncateweight(pstr, max_weight) && return true
    PauliPropagation.truncatefrequency(coeff, max_freq) && return true
    PauliPropagation.truncatesins(coeff, max_sins) && return true
    !isnothing(customtruncfunc) && customtruncfunc(pstr, coeff) && return true
    return false
end

"""
    applymergetruncate!(gate::PauliRotation, prop_cache::VectorPauliPropagationCache, theta; fused::Bool=false, kwargs...)

Fused, task-partitioned overload of `applymergetruncate!` for `PauliRotation` -- see file header.
Only used when `fused=true`; otherwise falls through (via `invoke`) to stock behavior.
"""
function PauliPropagation.applymergetruncate!(gate::PauliPropagation.PauliRotation, prop_cache::PauliPropagation.VectorPauliPropagationCache, theta;
    fused::Bool=false,
    min_abs_coeff::Real=1e-10, max_weight::Real=Inf, max_freq::Real=Inf, max_sins::Real=Inf, customtruncfunc=nothing,
    thread::Bool=true, kwargs...)

    # invoke function from library
    if !fused
        return invoke(PauliPropagation.applymergetruncate!,
            Tuple{PauliPropagation.PauliRotation,PauliPropagation.AbstractPauliPropagationCache,typeof(theta)},
            gate, prop_cache, theta;
            min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc, thread, kwargs...)
    end

    PauliPropagation._check_qind_range(PauliPropagation.nqubits(prop_cache), gate.qinds)

    if PauliPropagation.activesize(prop_cache) == 0
        return prop_cache
    end

    gate_mask = PauliPropagation.symboltoint(PauliPropagation.paulitype(prop_cache), gate.symbols, gate.qinds)

    truncfunc(pstr, coeff) = _fusedtruncfunc(pstr, coeff; min_abs_coeff, max_weight, max_freq, max_sins, customtruncfunc)

    # truncats during application of the gate to 1. make merge!() faster and 2. save on the extra truncation!() pass
    _fusedapplytruncatepaulirotation!(prop_cache, gate_mask, theta, truncfunc; thread)

    PauliPropagation.merge!(prop_cache; thread, truncfunc, kwargs...)

    return prop_cache
end

function _fusedapplytruncatepaulirotation!(prop_cache::PauliPropagation.VectorPauliPropagationCache, gate_mask::TT, theta, truncfunc;
    thread::Bool=true) where {TT}

    n_old = activesize(prop_cache)

    # Only the first `old_sortedprefix` elements are guaranteed sorted going in (some gates, e.g.
    # CliffordGate, reset it without merging). Compaction is order-preserving, so kept survivors
    # from that sorted prefix stay contiguous at the front of the output -- counting them below
    # gives the new sortedprefix directly, with no separate re-derivation needed.
    old_sortedprefix = sortedprefix(mainsum(prop_cache))

    cos_val = cos(theta)
    sin_val = sin(theta)

    task_partitioner = AK.TaskPartitioner(n_old, PauliPropagation.maxtasks(thread), PropagationBase._MIN_ELEMS_PER_TASK)
    n_tasks = task_partitioner.num_tasks

    terms_full = terms(mainsum(prop_cache))
    coeffs_full = coeffs(mainsum(prop_cache))
    aux_terms = terms(auxsum(prop_cache))
    aux_coeffs = coeffs(auxsum(prop_cache))

    kept_counts = Vector{Int}(undef, n_tasks)
    new_counts = Vector{Int}(undef, n_tasks)
    sorted_kept_counts = Vector{Int}(undef, n_tasks)

    # dry run: each task counts its own kept (commuting or surviving cos-branch) and new (surviving
    # sin-branch) output sizes, without writing
    if n_tasks == 1
        kept_counts[1], new_counts[1], sorted_kept_counts[1] = _fusedbranchwrite!(aux_terms, aux_coeffs, 1, aux_terms, aux_coeffs, 1,
            terms_full, coeffs_full, 1, n_old, gate_mask, cos_val, sin_val, truncfunc, old_sortedprefix, Val(false))
    else
        AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
            rng = task_partitioner[task_id]
            kept_counts[task_id], new_counts[task_id], sorted_kept_counts[task_id] = _fusedbranchwrite!(aux_terms, aux_coeffs, 1, aux_terms, aux_coeffs, 1,
                terms_full, coeffs_full, rng.start, rng.stop, gate_mask, cos_val, sin_val, truncfunc, old_sortedprefix, Val(false))
        end
    end

    # small serial prefix sums over just the per-task counts (mirrors sortedtailmerge!'s offset bookkeeping)
    kept_offsets = Vector{Int}(undef, n_tasks + 1)
    new_offsets = Vector{Int}(undef, n_tasks + 1)
    kept_offsets[1] = 1
    new_offsets[1] = 1
    @inbounds for t in 1:n_tasks
        kept_offsets[t+1] = kept_offsets[t] + kept_counts[t]
        new_offsets[t+1] = new_offsets[t] + new_counts[t]
    end
    n_kept = kept_offsets[n_tasks+1] - 1
    n_new = new_offsets[n_tasks+1] - 1
    n_total = n_kept + n_new
    new_sortedprefix = sum(sorted_kept_counts)

    resize_factor = 1.5
    if PauliPropagation.capacity(prop_cache) < n_total
        resize!(prop_cache, round(Int, n_total * resize_factor))
        terms_full = terms(mainsum(prop_cache))
        coeffs_full = coefficients(mainsum(prop_cache))
        aux_terms = terms(auxsum(prop_cache))
        aux_coeffs = coefficients(auxsum(prop_cache))
    end

    # real pass: redo the same walk, now writing each task's output directly into its final
    # position -- kept head at kept_offsets[task], new tail right after it at n_kept+new_offsets[task]
    if n_tasks == 1
        _fusedbranchwrite!(aux_terms, aux_coeffs, kept_offsets[1], aux_terms, aux_coeffs, n_kept + new_offsets[1],
            terms_full, coeffs_full, 1, n_old, gate_mask, cos_val, sin_val, truncfunc, old_sortedprefix, Val(true))
    else
        AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
            rng = task_partitioner[task_id]
            _fusedbranchwrite!(aux_terms, aux_coeffs, kept_offsets[task_id], aux_terms, aux_coeffs, n_kept + new_offsets[task_id],
                terms_full, coeffs_full, rng.start, rng.stop, gate_mask, cos_val, sin_val, truncfunc, old_sortedprefix, Val(true))
        end
    end

    swapsums!(prop_cache)
    setactivesize!(prop_cache, n_total)
    setsortedprefix!(mainsum(prop_cache), new_sortedprefix)

    return prop_cache
end

# Walks terms[lo:hi], branching each term on (anti)commutation with gate_mask and truncating
# inline. Writes survivors from kept_start/new_start when DoWrite; otherwise only counts (dry-run
# sizing pass). n_sorted_kept counts survivors that originated within the old sorted prefix, which
# stay contiguous at the front of the kept head -- see caller. Returns (n_kept, n_new, n_sorted_kept).
@inline function _fusedbranchwrite!(kept_out_terms, kept_out_coeffs, kept_start,
    new_out_terms, new_out_coeffs, new_start,
    terms, coeffs, lo, hi, gate_mask::TT, cos_val, sin_val, truncfunc::F, old_sortedprefix::Int, ::Val{DoWrite}) where {TT,F,DoWrite}

    kept_pos = kept_start
    n_sorted_kept = 0
    new_pos = new_start

    @inbounds for ii in lo:hi
        pstr = terms[ii]
        coeff = coeffs[ii]

        if PauliPropagation.commutes(gate_mask, pstr)
            if DoWrite
                kept_out_terms[kept_pos] = pstr
                kept_out_coeffs[kept_pos] = coeff
            end
            kept_pos += 1
            ii <= old_sortedprefix && (n_sorted_kept += 1)
        else
            coeff1 = coeff * cos_val
            if !truncfunc(pstr, coeff1)
                if DoWrite
                    kept_out_terms[kept_pos] = pstr
                    kept_out_coeffs[kept_pos] = coeff1
                end
                kept_pos += 1
                ii <= old_sortedprefix && (n_sorted_kept += 1)
            end

            new_pstr, sign = PauliPropagation.paulirotationproduct(gate_mask, pstr)
            coeff2 = coeff * sin_val * sign
            if !truncfunc(new_pstr, coeff2)
                if DoWrite
                    new_out_terms[new_pos] = new_pstr
                    new_out_coeffs[new_pos] = coeff2
                end
                new_pos += 1
            end
        end
    end

    return (kept_pos - kept_start, new_pos - new_start, n_sorted_kept)
end
