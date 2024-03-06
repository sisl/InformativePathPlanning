mutable struct GaussianProcess
    m           # mean
    mXq         # mean function at query points
    k           # covariance function
    X           # design points
    X_query     # query points (assuming these always stay the same)
    y           # objective values
    ν           # noise variance
    KXX         # K(X,X) the points we have measured
    inv_KXX_ν   # inv(K(X,X) + νI)
    KXqX        # K(Xq,X) the points we are querying and we have measured
    KXqXq
end

μ(X, m) = [m(x) for x in X]
# μ(X::Vector{Int64}, m) = reshape([m(x) for x in X][1], length(X))
# μ(X::Vector{Vector{Int64}}, m) = reshape([m(x) for x in X][1], length(X))
# μ(X, m::Interpolations.Extrapolation{Float64, 2, ScaledInterpolation{Float64, 2, Interpolations.BSplineInterpolation{Float64, 2, Matrix{Float64}, BSpline{Linear{Throw{OnGrid}}}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}, BSpline{Linear{Throw{OnGrid}}}, Tuple{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}}, BSpline{Linear{Throw{OnGrid}}}, Throw{Nothing}}) = reshape([m(x) for x in X][1], length(X))

# Σ(X, k) = [k(x,x′) for x in X, x′ in X]
# K(X, X′, k) = [k(x,x′) for x in X, x′ in X′]
Σ(X, k) = kernelmatrix(k, X, X)
K(X, X′, k) = kernelmatrix(k, X, X′)
# ν(X, K) = [variance(x, X, K) for x in X]

function mvnrand(rng, μ, Σ, inflation=1e-6)
    N = MvNormal(μ, Σ + inflation*I)
    #N = MvNormal(μ, Σ)
    return rand(rng, N)
end
Base.rand(rng, GP, X) = mvnrand(rng, μ(X, GP.m), Σ(X, GP.k))
Base.rand(rng, GP, μ_calc, Σ_calc) = mvnrand(rng, μ_calc, Σ_calc)

function query_no_data(GP::GaussianProcess)
    μₚ = GP.mXq
    S = GP.KXqXq
    νₚ = diag(S) .+ 1e-4 # eps prevents numerical issues

    y_min = Inf
    EI = [expected_improvement_cgp(y_min, μₚ[i], sqrt.(νₚ[i])) for i in 1:length(μₚ)]
    lcb = [LB(μₚ[i], sqrt.(νₚ[i])) for i in 1:length(μₚ)]
    return (μₚ, νₚ, S, EI, lcb)
end

function schur_inverse!(inv_A::Matrix{Float64}, A::Matrix{Float64}, B::Matrix{Float64}, C::Matrix{Float64}, D::Matrix{Float64})
    schur_complement = reshape(D - C*inv_A*B, size(D))
    inv_sc = inv(schur_complement)
    inv_M = [inv_A + inv_A*B*inv_sc*C*inv_A  -inv_A*B*inv_sc;
              -inv_sc*C*inv_A                 inv_sc]

    return inv_M
end

function query(GP::GaussianProcess)
    tmp = GP.KXqX * GP.inv_KXX_ν #/ (GP.KXX + Diagonal(GP.ν .+ 1e-4))
    μₚ = GP.mXq + tmp*(GP.y - μ(GP.X, GP.m))
    S = GP.KXqXq - tmp*GP.KXqX'
    νₚ = diag(S) .+ 1e-4 # eps prevents numerical issues

    y_min = minimum(GP.y)
    EI = [expected_improvement_cgp(y_min, μₚ[i], sqrt.(νₚ[i])) for i in 1:length(μₚ)]
    lcb = [LB(μₚ[i], sqrt.(νₚ[i])) for i in 1:length(μₚ)]
    return (μₚ, νₚ, S, EI, lcb)
end

function posterior(GP::GaussianProcess, X_samp, y_samp, ν_samp)
    if GP.X == []

        KXX = kernelmatrix(GP.k, X_samp, X_samp)
        KXqX = kernelmatrix(GP.k, GP.X_query, X_samp)
        inv_KXX_ν = inv(KXX + Diagonal(ν_samp .+ 1e-4))

        return GaussianProcess(GP.m, GP.mXq, GP.k, X_samp, GP.X_query, y_samp, ν_samp, KXX, inv_KXX_ν, KXqX, GP.KXqXq)
    else
        a = kernelmatrix(GP.k, GP.X, X_samp)
        aT = Matrix(a')
        KXsampXsamp = kernelmatrix(GP.k, X_samp, X_samp)

        inv_KXX_ν = schur_inverse!(GP.inv_KXX_ν, GP.KXX + Diagonal(GP.ν .+ 1e-4), a, aT, KXsampXsamp + Diagonal(ν_samp .+ 1e-4)) 
        
        KXX = [GP.KXX a; aT KXsampXsamp] #KXX = [GP.KXX a; a' I]
        KXqX = [GP.KXqX kernelmatrix(GP.k, GP.X_query, X_samp)]#hcat(GP.KXqX, kernelmatrix(k, GP.X_query, X_samp))

        # @show inv(KXX + Diagonal([GP.ν; ν_samp] .+ 1e-4))
        # @show inv_KXX_ν
        # @show isapprox(inv_KXX_ν, inv(KXX + Diagonal([GP.ν; ν_samp] .+ 1e-4)))

        return GaussianProcess(GP.m, GP.mXq, GP.k, [GP.X; X_samp], GP.X_query, [GP.y; y_samp], [GP.ν; ν_samp], KXX, inv_KXX_ν, KXqX, GP.KXqXq)
    end
end

function expected_improvement_cgp(y_min, μ, σ)
    p_imp = cdf(Normal(μ, σ), y_min) 
    p_ymin = pdf(Normal(μ, σ), y_min)
    return (y_min - μ)*p_imp + σ^2*p_ymin
end

# Lower Confidence Bound
function LB(μ, σ, α=1.0)
    return μ - α*σ 
end