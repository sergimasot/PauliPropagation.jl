###
##
# Sort-tail-and-merge: a cheaper alternative to a full re-sort for merging a propagation cache
# whose active terms are [sorted, duplicate-free head][unsorted tail]. The tail may itself contain
# duplicates (e.g. when n_old is a sortedprefix() smaller than the true previous active size,
# because an intervening gate invalidated it without merging) -- both head-tail and intra-tail
# collisions are merged via mergefunc. Gate-agnostic; callers just need the head to be sorted.
##
###

# Below this many head elements, a single sequential task is used regardless of thread count --
# avoids paying task-spawn/partitioning overhead while the head is still small.
const _TAILMERGE_MIN_ELEMS_PER_TASK = 1024

# _merge!() only dispatches to sortedtailmerge! when:
# sortedprefix(term_sum) / length(term_sum) > _TAILMERGE_SORTEDPREFIX_FRACTION
# below it, a full re-sort is cheaper.
const _TAILMERGE_SORTEDPREFIX_FRACTION = 0.4

"""
    sortedtailmerge!(prop_cache::AbstractPropagationCache, n_old::Int, n_new::Int; thread::Bool=true)

Merge the sorted, duplicate-free head `[1:n_old]` of `mainsum(prop_cache)`'s active terms against
its unsorted tail `[n_old+1:n_new]`, in place of a full re-sort of `[1:n_new]`. The tail may contain
duplicates, within itself or against the head; all are combined via `mergefunc`. Updates
`activesize`/`sortedprefix` on the result. Set `thread=false` to force sequential execution.
"""
function sortedtailmerge!(prop_cache::AbstractPropagationCache, n_old::Int, n_new::Int; thread::Bool=true)
    n_tail = n_new - n_old
    if n_tail == 0
        setsortedprefix!(mainsum(prop_cache), n_old)
        return prop_cache
    end

    active_terms = terms(mainsum(prop_cache))
    active_coeffs = coefficients(mainsum(prop_cache))

    # sort just the tail into scratch buffers, small relative to n_old for a growing circuit
    tail_terms = active_terms[n_old+1:n_new]
    tail_coeffs = active_coeffs[n_old+1:n_new]
    tail_perm = collect(1:n_tail)
    AK.sortperm!(tail_perm, tail_terms)
    tail_terms = tail_terms[tail_perm]
    tail_coeffs = tail_coeffs[tail_perm]

    # merge output goes into the auxiliary sum, swapped in as the new active one at the end
    aux_terms = terms(auxsum(prop_cache))
    aux_coeffs = coefficients(auxsum(prop_cache))

    task_partitioner = AK.TaskPartitioner(n_old, _maxtasks(thread), _TAILMERGE_MIN_ELEMS_PER_TASK)
    n_tasks = task_partitioner.num_tasks

    if n_tasks == 1
        merged_count = _tailmerge_write!(aux_terms, aux_coeffs, 1,
            active_terms, active_coeffs, 1, n_old, tail_terms, tail_coeffs, 1, n_tail, Val(true))
        swapsums!(prop_cache)
        setactivesize!(prop_cache, merged_count)
        setsortedprefix!(mainsum(prop_cache), merged_count)
        return prop_cache
    end

    # match each head chunk to a tail slice by binary-searching its largest term into the
    # (already-sorted) tail, so each (head-chunk, tail-slice) pair can merge independently
    tail_bounds_per_task = Vector{Int}(undef, n_tasks + 1)
    tail_bounds_per_task[1] = 1
    tail_bounds_per_task[n_tasks+1] = n_tail + 1
    @inbounds for task_id in 1:(n_tasks-1)
        head_chunk_boundary_term = active_terms[task_partitioner[task_id].stop]
        tail_bounds_per_task[task_id+1] = searchsortedlast(tail_terms, head_chunk_boundary_term) + 1
    end

    # dry run: each task counts its own merged output size (unknown ahead of time due to collisions)
    merged_counts_per_task = Vector{Int}(undef, n_tasks)
    AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
        head_range = task_partitioner[task_id]
        merged_counts_per_task[task_id] = _tailmerge_write!(aux_terms, aux_coeffs, 1,
            active_terms, active_coeffs, head_range.start, head_range.stop,
            tail_terms, tail_coeffs, tail_bounds_per_task[task_id], tail_bounds_per_task[task_id+1] - 1, Val(false))
    end

    # prefix sum over the per-task counts gives each task its exact final write offset
    write_offsets_per_task = Vector{Int}(undef, n_tasks + 1)
    write_offsets_per_task[1] = 1
    @inbounds for task_id in 1:n_tasks
        write_offsets_per_task[task_id+1] = write_offsets_per_task[task_id] + merged_counts_per_task[task_id]
    end
    total_merged_count = write_offsets_per_task[n_tasks+1] - 1

    # real pass: each task redoes the same merge, now writing directly into its final position
    AK.itask_partition(n_tasks, n_tasks, 1) do task_id, _
        head_range = task_partitioner[task_id]
        _tailmerge_write!(aux_terms, aux_coeffs, write_offsets_per_task[task_id],
            active_terms, active_coeffs, head_range.start, head_range.stop,
            tail_terms, tail_coeffs, tail_bounds_per_task[task_id], tail_bounds_per_task[task_id+1] - 1, Val(true))
    end

    swapsums!(prop_cache)
    setactivesize!(prop_cache, total_merged_count)
    setsortedprefix!(mainsum(prop_cache), total_merged_count)

    return prop_cache
end

# Two-pointer merge of a sorted, duplicate-free head slice against a sorted tail slice that may
# itself contain runs of duplicates (both against the head and within itself); all are combined via
# mergefunc. Writes into out_terms/out_coeffs from out_start when DoWrite; otherwise only counts
# (the dry-run sizing pass). Returns the number of output elements.
@inline function _tailmerge_write!(out_terms, out_coeffs, out_start,
    head_terms, head_coeffs, head_lo, head_hi,
    tail_terms, tail_coeffs, tail_lo, tail_hi,
    ::Val{DoWrite}) where {DoWrite}

    head_i = head_lo
    tail_j = tail_lo
    write_pos = out_start
    @inbounds while head_i <= head_hi && tail_j <= tail_hi
        head_term = head_terms[head_i]
        tail_term = tail_terms[tail_j]
        if head_term == tail_term
            # collision: merge the head term with the *entire run* of equal tail terms
            merged_coeff = head_coeffs[head_i]
            while tail_j <= tail_hi && tail_terms[tail_j] == tail_term
                merged_coeff = mergefunc(merged_coeff, tail_coeffs[tail_j])
                tail_j += 1
            end
            if DoWrite
                out_terms[write_pos] = head_term
                out_coeffs[write_pos] = merged_coeff
            end
            write_pos += 1
            head_i += 1
        elseif head_term < tail_term
            if DoWrite
                out_terms[write_pos] = head_term
                out_coeffs[write_pos] = head_coeffs[head_i]
            end
            write_pos += 1
            head_i += 1
        else
            # tail term has no match in the head (yet): merge its own run of duplicates first
            merged_coeff = tail_coeffs[tail_j]
            tail_j += 1
            while tail_j <= tail_hi && tail_terms[tail_j] == tail_term
                merged_coeff = mergefunc(merged_coeff, tail_coeffs[tail_j])
                tail_j += 1
            end
            if DoWrite
                out_terms[write_pos] = tail_term
                out_coeffs[write_pos] = merged_coeff
            end
            write_pos += 1
        end
    end
    @inbounds while head_i <= head_hi
        if DoWrite
            out_terms[write_pos] = head_terms[head_i]
            out_coeffs[write_pos] = head_coeffs[head_i]
        end
        write_pos += 1
        head_i += 1
    end
    @inbounds while tail_j <= tail_hi
        tail_term = tail_terms[tail_j]
        merged_coeff = tail_coeffs[tail_j]
        tail_j += 1
        while tail_j <= tail_hi && tail_terms[tail_j] == tail_term
            merged_coeff = mergefunc(merged_coeff, tail_coeffs[tail_j])
            tail_j += 1
        end
        if DoWrite
            out_terms[write_pos] = tail_term
            out_coeffs[write_pos] = merged_coeff
        end
        write_pos += 1
    end

    return write_pos - out_start
end
