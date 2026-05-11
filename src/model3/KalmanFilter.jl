using LinearAlgebra
using Statistics
using Distributions

## Kalman Filter and RTS Smoother
# Model:
# z = Normal(Az,Q)
# x = Normal(Cz, R)

function kalman_filter_smoother(xs::AbstractMatrix{S}; params, stateonly=false, EM=false) where S <: AbstractFloat
    # Extracting parameters

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

    # Terms associated with Kalman gain

    δt    = Array{S}(undef, N)
    S_t   = Array{S}(undef, N, N)
    KtS_t = Array{S}(undef, D, N) # stands for "K times S_t"
    # K_t = KtS_t / S_t

    #  ------ initialization ------
    # predict μ_1_0
    μ_t_tmin1[:,    1] .= A * μ0
    Σ_t_tmin1[:, :, 1] .= A * Σ0 * A' + Q

    # update μ_1_1
    S_t   .= (C * Σ_t_tmin1[:, :, 1] * C' + R)
    KtS_t .= Σ_t_tmin1[:, :, 1] * C'

    δt    .= xs[:, 1] - C * μ_t_tmin1[:, 1]

    μ_t_t[:,    1] .= μ_t_tmin1[:, 1] + KtS_t * (S_t \ δt)
    Σ_t_t[:, :, 1] .= (I - KtS_t * (S_t \ C)) * Σ_t_tmin1[:, :, 1]

    # ------ iterative loop ------

    for t in 2:T
        # Predict step
        # μ_2_1
        μ_t_tmin1[:,    t] .= A * μ_t_t[:, t-1]
        Σ_t_tmin1[:, :, t] .= A * Σ_t_t[:, :, t-1] * A' + Q

        # Update step
        # μ_2_2
        S_t   .= (C * Σ_t_tmin1[:, :, t] * C' + R)
        KtS_t .= Σ_t_tmin1[:, :, t] * C'

        δt    .= xs[:, t] - C * μ_t_tmin1[:, t]

        μ_t_t[:,    t] .= μ_t_tmin1[:, t] + KtS_t * (S_t \ δt)
        Σ_t_t[:, :, t] .= (I - KtS_t * (S_t \ C)) * Σ_t_tmin1[:, :, t]
    end

    # RTS smoother

    μ_t_T[:,    T] .= μ_t_t[:,    T]
    Σ_t_T[:, :, T] .= Σ_t_t[:, :, T]

    # Terms associated with gain

    μ_t_T = Array{S}(undef, D, T)
    Σ_t_T = Array{S}(undef, D, D, T)

    G_t   = Array{S}(undef, D, D)

    # ------ loop ------

    for t in T-1:-1:1
        G_t .= Σ_t_t[:, :, t] * A' / Σ_t_tmin1[:, :, t+1]

        μ_t_T[:,    t] .= μ_t_t[:,    t] + G_t * (μ_t_T[:, t+1] - μ_t_tmin1[:, t+1])
        Σ_t_T[:, :, t] .= Σ_t_t[:, :, t] + G_t * (Σ_t_T[:, :, t+1] - Σ_t_tmin1[:, :, t+1]) * G_t'
    end

    # Correcting Priors

    G_0   = Σ0 * A' / Σ_t_tmin1[:, :, 1]
    μ_0_T = μ0 + G_0 * (μ_t_T[:, 1] - μ_t_tmin1[:, 1])
    Σ_0_T = Σ0 + G_0 * (Σ_t_T[:, :, 1] - Σ_t_tmin1[:, :, 1]) * G_0'

    if stateonly
        return μ_t_T
    elseif EM
        return (; μ_t_T, Σ_t_T, μ_0_T, Σ_0_T, μ_t_t, Σ_t_t, μ_t_tmin1, Σ_t_tmin1)
    else
        return (; μ_t_T, Σ_t_T, μ_0_T, Σ_0_T)
    end
end
