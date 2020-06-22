module NumericalIntegration

using LinearAlgebra
using Logging
using Base.Iterators
using Interpolations

export integrate, cumul_integrate
export Trapezoidal, TrapezoidalEven, TrapezoidalFast, TrapezoidalEvenFast
export SimpsonEven, SimpsonEvenFast
export RombergEven
export IntegrationMethod

abstract type IntegrationMethod end

struct Trapezoidal         <: IntegrationMethod end
struct TrapezoidalEven     <: IntegrationMethod end
struct TrapezoidalFast     <: IntegrationMethod end
struct TrapezoidalEvenFast <: IntegrationMethod end
struct SimpsonEven         <: IntegrationMethod end # https://en.wikipedia.org/wiki/Simpson%27s_rule#Alternative_extended_Simpson.27s_rule
struct SimpsonEvenFast     <: IntegrationMethod end
struct RombergEven{T<:AbstractFloat} <: IntegrationMethod
    acc::T
end # https://en.wikipedia.org/wiki/Romberg%27s_method
RombergEven() = RombergEven(1e-12)

const HALF = 1//2

#documentation

"""
    integrate(x,y...)

Compute numerical integral of y(x) from x=x[1] to x=x[end]. Return a scalar of the same type as the input. If no method is supplied, use Trapezdoial.
"""
function integrate(x,y...) end


"""
    cumul_integrate(x,y...)

Compute cumulative numerical integral of y(x) from x=x[1] to x=x[end]. Return a vector with elements of the same type as the input. If no method is supplied, use Trapezdoial.
"""
function cumul_integrate end

#implementation

"""
    integrate(x::AbstractVector, y::AbstractVector, ::Trapezoidal)

Use Trapezoidal rule. This is the default when no method is supplied.
"""
function integrate(x::AbstractVector, y::AbstractVector, ::Trapezoidal)
    @assert axes(x) == axes(y) "x and y must have the same geometry"
    @assert length(x) ≥ 2 "vectors must contain at least two elements"

    return integrate(x, y, TrapezoidalFast())
end
function integrate(x::AbstractRange, y::AbstractVector, ::Trapezoidal)
    return integrate(x, y, TrapezoidalEven())
end

"""
    integrate(x::AbstractVector, y::AbstractVector, ::TrapezoidalEven)

Use Trapezoidal rule, assuming evenly spaced vector x.
"""
function integrate(x::AbstractVector, y::AbstractVector, ::TrapezoidalEven)
    @assert axes(x) == axes(y) "x and y must have the same geometry"
    @assert length(x) ≥ 2 "vectors must contain at least two elements"

    return integrate(x, y, TrapezoidalEvenFast())
end

"""
    integrate(x::AbstractVector, y::AbstractVector, ::TrapezoidalFast)

Use Trapezoidal rule. Unsafe method: no bound checking.
"""
function integrate(x::AbstractVector, y::AbstractVector, ::TrapezoidalFast)
    @inbounds retval = (x[begin+1] - x[begin]) * (y[begin] + y[begin+1])
    @inbounds @fastmath @simd for i in eachindex(y)[begin+1:end-1]
        retval += (x[i+1] - x[i]) * (y[i] + y[i+1])
    end
    return HALF * retval
end
function integrate(x::AbstractRange, y::AbstractVector, ::TrapezoidalFast)
    return integrate(x, y, TrapezoidalEvenFast())
end

"""
    integrate(x::AbstractVector, y::AbstractVector, ::TrapezoidalEvenFast)

Use Trapezoidal rule, assuming evenly spaced vector x. Unsafe method: no bound checking.
"""
function integrate(x::AbstractVector, y::AbstractVector, ::TrapezoidalEvenFast)
    if length(x) < 3
        @inbounds return HALF * (x[begin+1] - x[begin]) * (y[begin] + y[begin+1])
    else
        @inbounds retval = y[begin+1]
        @inbounds @fastmath @simd for i in eachindex(y)[begin+2:end-1]
            retval += y[i]
        end
        @inbounds return (x[begin+1] - x[begin]) * (retval + HALF * (y[begin] + y[end]))
    end
end

"""
    integrate(x::AbstractVector, y::AbstractVector, ::SimpsonEven)

Use Simpson's rule, assuming evenly spaced vector x.
"""
function integrate(x::AbstractVector, y::AbstractVector, ::SimpsonEven)
    @assert axes(x) == axes(y) "x and y must have the same geometry"
    @assert length(x) ≥ 4 "vectors must contain at least 4 elements"

    return integrate(x, y, SimpsonEvenFast())
end

"""
    integrate(x::AbstractVector, y::AbstractVector, ::SimpsonEven)

Use Simpson's rule, assuming evenly spaced vector x.  Unsafe method: no bound checking.
"""
function integrate(x::AbstractVector, y::AbstractVector, ::SimpsonEvenFast)
    @inbounds retval = (17 * (y[begin] + y[end]) + 59 * (y[begin+1] + y[end-1]) +
                        43 * (y[begin+2] + y[end-2]) + 49 * (y[begin+3] + y[end-3])) / 48
    @fastmath @inbounds for i in eachindex(y)[begin+4:end-4]
        retval += y[i]
    end
    @inbounds return (x[begin+1] - x[begin]) * retval
end

"""
    integrate(x::AbstractVector, y::AbstractMatrix, method; dims=2)

When y is an array, compute integral along dimension specified by dims (default 2: columns).
"""
function integrate(x::AbstractVector, y::AbstractMatrix, M::IntegrationMethod; dims=2)
    out = [integrate(x,selectdim(y,dims,j),M) for j=axes(y,dims)]
    return out
end

function integrate(x::AbstractVector, y::AbstractVector, m::RombergEven)
    @assert axes(x) == axes(y) "x and y must have the same geometry"
    @assert ((length(x) - 1) & (length(x) - 2)) == 0 "Need length of vector to be 2^n + 1"
    maxsteps::Integer = Int(log2(length(x)-1))
    @assert firstindex(x) == 1 "RombergEven requires 1:N indexed vectors."
    @inbounds h = x[end] - x[1]
    @inbounds v = (y[1] + y[end]) * h * HALF
    rombaux = Matrix{typeof(v)}(undef, maxsteps, 2)
    rombaux[1,1] = v
    prevcol = 1
    currcol = 2
    # Precomputed values for norm check
    acc_compare = m.acc * oneunit(norm(v,Inf))
    minsteps = maxsteps ÷ 3
    @inbounds for i in 1 : (maxsteps-1)
        h *= HALF
        npoints = 1 << (i-1)
        jumpsize = div(length(x)-1, 2*npoints)
        c = zero(eltype(y))
        for j in 1 : npoints
            c += y[1 + (2*j-1)*jumpsize]
        end
        rombaux[1, currcol] = h*c + HALF*rombaux[1, prevcol]
        for j in 2 : (i+1)
            n_k = 4^(j-1)
            rombaux[j, currcol] = (n_k*rombaux[j-1, currcol] - rombaux[j-1, prevcol])/(n_k - 1)
        end

        # Clunky way to address Unitful compatibility, while also allowing for vectors
        normval = norm(rombaux[i, prevcol] - rombaux[i+1, currcol], Inf)
        if i > minsteps && normval < acc_compare
            return rombaux[i+1, currcol]
        end

        prevcol, currcol = currcol, prevcol
    end
    finalerr = norm(rombaux[maxsteps-1, currcol] - rombaux[maxsteps, prevcol], Inf)
    @warn "RombergEven :: final step reached, but accuracy not: $finalerr > $(m.acc)"
    @inbounds return rombaux[maxsteps, prevcol]
end


@inline function _midpoints(x::AbstractVector{T}) where T
    length(x) == 1 && return x
    return HALF * (x[begin:end-1] + x[begin+1:end])
end
@inline function _midpoints(x::AbstractRange)
    length(x) == 1 && return x
    Δx = HALF*step(x)
    return range(first(x)+Δx, stop=last(x)-Δx, length=length(x)-1)
end

function integrate(X::NTuple{N,AbstractVector}, Y::AbstractArray{T,N}, ::Trapezoidal) where {T,N}
    @assert length.(X) == size(Y)
    return integrate(X, Y, TrapezoidalFast())
end

function integrate(X::NTuple{N,AbstractVector}, Y::AbstractArray{T,N}, ::TrapezoidalFast) where {T,N}
    midpnts = map(_midpoints, X)
    Δ(x::AbstractVector) = length(x) > 1 ? diff(x) : 1
    Δxs = map(Δ, X)
    interp = LinearInterpolation(X,Y)
    f((Δx,x)) = prod(Δx)*interp(x...)
    return sum(f, zip(product(Δxs...), product(midpnts...)))
end

function integrate(X::NTuple{N,AbstractVector}, Y::AbstractArray{T,N}, ::TrapezoidalEvenFast) where {T,N}
    midpnts = map(_midpoints, X)
    Δ(x::AbstractVector) = length(x) > 1 ? x[2] - x[1] : 1
    Δx = prod(Δ, X)
    interp = LinearInterpolation(X,Y)
    f(x) = interp(x...)
    return Δx*sum(f, product(midpnts...))
end

function integrate(X::NTuple{N,AbstractRange}, Y::AbstractArray{T,N}, ::TrapezoidalEvenFast) where {T,N}
    midpnts = map(_midpoints, X)
    Δ(x::AbstractVector) = length(x) > 1 ? step(x) : 1
    Δx = prod(Δ, X)
    interp = LinearInterpolation(X,Y)
    f(x) = interp(x...)
    return Δx*sum(f, product(midpnts...))
end




#function integrate(X::Tuple{AbstractVector}, Y::AbstractVector{T}, M::IntegrationMethod) :: T where {T}
#    return integrate(X[1], Y, M)
#end
#"""
#    integrate(X::NTuple{N,AbstractVector}, Y::AbstractArray{T,N}, method, cache=nothing)
#
#Given an n-dimensional grid of values, compute the total integral along each dim
#"""
#function integrate(X::NTuple{N,AbstractVector}, Y::AbstractArray{T,N}, M::IntegrationMethod) :: T where {T,N}
#    n = last(size(Y))
#    cache = Vector{T}(undef, n)
#    x = X[1:N-1]
#    @inbounds for i in 1:n
#        cache[i] = integrate(x, selectdim(Y,N,i), M)
#    end
#    return integrate(X[end], cache, M)
#end



# cumulative integrals

"""
    cumul_integrate(x::AbstractVector, y::AbstractVector, ::Trapezoidal)

Use Trapezoidal rule. This is the default when no method is supplied.
"""
function cumul_integrate(x::AbstractVector, y::AbstractVector, ::Trapezoidal)
    @assert axes(x) == axes(y) "x and y must have the same geometry"
    @assert length(x) ≥ 2 "vectors must contain at least two elements"

    return cumul_integrate(x, y, TrapezoidalFast())
end
function cumul_integrate(x::AbstractRange, y::AbstractVector, ::Trapezoidal)
    return cumul_integrate(x, y, TrapezoidalEven())
end

"""
    cumul_integrate(x::AbstractVector, y::AbstractVector, ::TrapezoidalEven)

Use Trapezoidal rule, assuming evenly spaced vector x.
"""
function cumul_integrate(x::AbstractVector, y::AbstractVector, ::TrapezoidalEven)
    @assert axes(x) == axes(y) "x and y must have the same geometry"
    @assert length(x) ≥ 2 "vectors must contain at least two elements"

    return cumul_integrate(x, y, TrapezoidalEvenFast())
end

"""
    cumul_integrate(x::AbstractVector, y::AbstractVector, ::TrapezoidalFast)

Use Trapezoidal rule. Unsafe method: no bound checking.
"""
function cumul_integrate(x::AbstractVector, y::AbstractVector, ::TrapezoidalFast)
    # compute initial value
    @inbounds init = (x[begin+1] - x[begin]) * (y[begin] + y[begin+1])
    @assert axes(x) == axes(y) "x and y must have the same geometry"
    retarr = similar(y, typeof(init))
    retarr[begin] = zero(init)
    retarr[begin+1] = init

    # for all other values
    @inbounds @fastmath for i in eachindex(retarr)[begin+2:end] # not sure if @simd can do anything here
        retarr[i] = retarr[i-1] + (x[i] - x[i-1]) * (y[i] + y[i-1])
    end

    return HALF * retarr
end

"""
    cumul_integrate(x::AbstractVector, y::AbstractVector, ::TrapezoidalEvenFast)

Use Trapezoidal rule, assuming evenly spaced vector x. Unsafe method: no bound checking.
"""
function cumul_integrate(x::AbstractVector, y::AbstractVector, ::TrapezoidalEvenFast)
    @assert axes(x) == axes(y) "x and y must have the same geometry"
    @inbounds init = y[begin] + y[begin+1]
    retarr = similar(y, typeof(init))
    retarr[begin] = zero(init)
    retarr[begin+1] = init

    # for all other values
    @inbounds @fastmath for i in eachindex(retarr)[begin+2:end]
        retarr[i] = retarr[i-1] + (y[i] + y[i-1])
    end
    @inbounds return (x[begin+1] - x[begin]) * HALF * retarr
end

"""
    cumul_integrate(x::AbstractVector, y::AbstractMatrix, method; dims=2)

When y is an array, compute integral along each dimension specified by dims (default 2: columns)
"""
function cumul_integrate(x::AbstractVector, y::AbstractMatrix, M::IntegrationMethod; dims=2)
    return hcat([cumul_integrate(x,selectdim(y,dims,j),M) for j=1:size(y,dims)]...)
end

#default behaviour
integrate(x::AbstractVector, y::AbstractVector) = integrate(x, y, Trapezoidal())

function integrate(x::AbstractVector, y::AbstractMatrix; kwargs...)
    return integrate(x, y, Trapezoidal(); kwargs...)
end

integrate(X::NTuple, Y::AbstractArray) = integrate(X, Y, Trapezoidal())

cumul_integrate(x::AbstractVector, y::AbstractVector) = cumul_integrate(x, y, Trapezoidal())

function cumul_integrate(x::AbstractVector, y::AbstractMatrix; kwargs...)
    return cumul_integrate(x, y, Trapezoidal(); kwargs...)
end

end # module
