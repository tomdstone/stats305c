## Model:
# z = Az + Normal(0,I)
# x = Normal(Cz, R)
using LinearAlgebra
using Statistics
using Distributions


## Kalman

function kalman_filter_smoother(xs::AbstractMatrix{S}; params, EM = true) where S <: AbstractFloat
    A = params.A
    C = params.C
    Q = params.Q
    R = params.R
    μ0 = params.μ0
    Σ0 = params.Σ0

    N, T = size(xs)

    D = size(params.A, 1)

    # Kalman filter

    μ_t_t     = Array{S}(undef, D, T)
    μ_t_tmin1 = Array{S}(undef, D, T)

    Σ_t_t     = Array{S}(undef, D, D, T)
    Σ_t_tmin1 = Array{S}(undef, D, D, T)

    # K_t = Array{S}(undef, D, N)
    δt      = Array{S}(undef, N)
    S_t     = Array{S}(undef, N, N)
    KdS_t   = Array{S}(undef, D, N)

    #  ------ initialization ------
    # predict μ_1_0
    μ_t_tmin1[:,    1] .= A * μ0
    Σ_t_tmin1[:, :, 1] .= A * Σ0 * A' + Q

    # update μ_1_1
    S_t     .= (C * Σ_t_tmin1[:, :, 1] * C' + R)
    KdS_t   .= Σ_t_tmin1[:, :, 1] * C'

    # K_t .= (Σ_t_tmin1[:, :, t] * C') / S_t

    δt      .= xs[:, 1] - C * μ_t_tmin1[:, 1]

    μ_t_t[:,    1]  .= μ_t_tmin1[:, 1] + KdS_t * (S_t \ δt)
    Σ_t_t[:, :, 1]  .= (I - KdS_t * (S_t \ C)) * Σ_t_tmin1[:, :, 1]

    # ------ iterative loop ------

    for t in 2:T
        # Predict step
        # μ_2_1
        μ_t_tmin1[:,    t]  .= A * μ_t_t[:, t-1]
        Σ_t_tmin1[:, :, t]  .= A * Σ_t_t[:, :, t-1] * A' + Q

        # Update step
        # μ_2_2
        S_t     .= (C * Σ_t_tmin1[:, :, t] * C' + R)
        KdS_t   .= Σ_t_tmin1[:, :, t] * C'

        # K_t .= (Σ_t_tmin1[:, :, t] * C') / S_t

        δt      .= xs[:, t] - C * μ_t_tmin1[:, t]

        μ_t_t[:,    t]  .= μ_t_tmin1[:, t] + KdS_t * (S_t \ δt)
        Σ_t_t[:, :, t]  .= (I - KdS_t * (S_t \ C)) * Σ_t_tmin1[:, :, t]
    end

    # RTS smoother

    μ_t_T   = Array{S}(undef, D, T)
    Σ_t_T   = Array{S}(undef, D, D, T)

    G_t     = Array{S}(undef, D, D)

    # ------ initialization ------

    μ_t_T[:,    T] .= μ_t_t[:,    T]
    Σ_t_T[:, :, T] .= Σ_t_t[:, :, T]

    # ------ loop ------

    for t in T-1:-1:1
        G_t .= Σ_t_t[:, :, t] * A' / Σ_t_tmin1[:, :, t+1]

        μ_t_T[:,    t] .= μ_t_t[:,    t] + G_t * (μ_t_T[:, t+1] - μ_t_tmin1[:, t+1])
        Σ_t_T[:, :, t] .= Σ_t_t[:, :, t] + G_t * (Σ_t_T[:, :, t+1] - Σ_t_tmin1[:, :, t+1]) * G_t'
    end

    G_0 = Σ0 * A' / Σ_t_tmin1[:, :, 1]
    μ_0_T = μ0 + G_0 * (μ_t_T[:, 1] - μ_t_tmin1[:, 1])
    Σ_0_T = Σ0 + G_0 * (Σ_t_T[:, :, 1] - Σ_t_tmin1[:, :, 1]) * G_0'

    if EM
        return (; μ_t_T, Σ_t_T, μ_0_T, Σ_0_T, μ_t_t, Σ_t_t, μ_t_tmin1, Σ_t_tmin1)
    else
        return (; μ_t_T, Σ_t_T, μ_0_T, Σ_0_T)
    end
end

## Testing using oscillator model

A = zeros(Float64, 4, 4)
A[1:2,1:2] .= 0.95 * [cospi(2 * 10 / 200) -sinpi(2* 10 / 200); sinpi(2 * 10 / 200) cospi(2*10*200)]
A[3:4,3:4] .= 0.99 * [cospi(2 * 60 / 200) -sinpi(2* 60 / 200); sinpi(2 * 60 / 200) cospi(2*60*200)]

Q = I(4)
C = [1. 0. 1. 0.]
R = [1.;;]
μ0 = zeros(Float64, 4)
Σ0 = I(4)


T = 10_000
zs = rand(MvNormal(zeros(4), Q), T)
zs[:, 1] .+= A * rand(MvNormal(μ0, Σ0))

for t in 2:T
    zs[:, t] .+= A * zs[:, t-1]
end

xs = C * zs + rand(MvNormal(zeros(1), R), T)

a = kalman_filter_smoother(xs, params = (; A, Q, C, R, μ0, Σ0), EM=true)
