tonumber(x::Number) = x

_invertfunc(func::F) where {F<:Function} = args -> !func(args...)

# thread=false runs everything on a single thread
maxtasks(thread::Bool) = thread ? Threads.nthreads() : 1

# for CPU-only code likeThreads.@spawn
# GPU extensions override this for their array for fallback functionality
_iscpuarray(::AbstractArray) = true

function _thrownotimplemented(::Type{T}, func_name::Symbol) where T
    error("Function '", func_name, "' not implemented for type '", T, "'.")
end

function _thrownotimplemented(::T, func_name::Symbol) where T
    _thrownotimplemented(T, func_name)
end