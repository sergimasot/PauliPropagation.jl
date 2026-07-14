module Performance
# This is a module for performance-oriented optimizations that
# 1) may eventually become the default, and/or
# 2) inherently change the output (slightly) for performance benefits
#
# Opt in via `Performance.propagate` and `Performance.propagate!`.

using PauliPropagation
using PauliPropagation.PropagationBase

using AcceleratedKernels
const AK = AcceleratedKernels

# PauliSum overload for PauliRotation
include("./fused_dict.jl")


# VectorPauliSum overload for PauliRotation
include("./fused_vector.jl")


"""
    propagate(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)

Like `PauliPropagation.propagate`, but defaults `fused=true` to use this module's fused
`applymergetruncate!` overloads. Pass `fused=false` for byte-identical stock results.
"""
function propagate(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)
    return PauliPropagation.propagate(circuit, thing, thetas; fused, kwargs...)
end

"""
    propagate!(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)

In-place counterpart of `propagate`. See `propagate` for details.
"""
function propagate!(circuit, thing, thetas=nothing; fused::Bool=true, kwargs...)
    return PauliPropagation.propagate!(circuit, thing, thetas; fused, kwargs...)
end

end