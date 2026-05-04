
function kfs_matmul(xs::AbstractMatrix{S}; params, EM = true) where S <: AbstractFloat
    A = params.A
    C = params.C
    Q = I
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

    init = copy(Σ_t_t)

    # K_t = Array{S}(undef, D, N)
    δt      = Array{S}(undef, N)
    S_t     = Array{S}(undef, N, N)
    KdS_t   = Array{S}(undef, D, N)

    #  ------ initialization ------

    # predict μ_1_0
    mul!(μ_t_tmin1[:, 1], A, μ0)
    Σ_t_tmin1[:, :, 1] .= muladd(A * Σ0, A', Q)

    # update μ_1_1
    S_t .= muladd(C * Σ_t_tmin1[:, :, 1], C', R)
    mul!(KdS_t, Σ_t_tmin1[:, :, 1], C')

    # K_t .= (Σ_t_tmin1[:, :, t] * C') / S_t

    δt .= muladd(C, μ_t_tmin1[:, 1], xs[:, 1])

    μ_t_t[:,    1]  .= muladd(KdS_t, S_t \ δt, μ_t_tmin1[:, 1])
    mul!(Σ_t_t[:, :, 1], muladd(-KdS_t, S_t \ C, I), Σ_t_tmin1[:, :, 1])

    # ------ iterative loop ------

    for t in 2:T
        # Predict step
        # μ_2_1
        mul!(μ_t_tmin1[:, t], A, μ_t_t[:, t-1])
        Σ_t_tmin1[:, :, 1] .= muladd(A, Σ_t_t[:, :, t-1] * A', Q)

        # μ_t_tmin1[:,    t]  .= A * μ_t_t[:, t-1]
        # Σ_t_tmin1[:, :, t]  .= A * Σ_t_t[:, :, t-1] * A' + Q

        # Update step
        # μ_2_2
        S_t .= muladd(C, Σ_t_tmin1[:, :, t] * C', R)
        mul!(KdS_t, Σ_t_tmin1[:, :, t], C')

        # K_t .= (Σ_t_tmin1[:, :, t] * C') / S_t

        δt .= muladd(C, -μ_t_tmin1[:, t], xs[:, t])

        μ_t_t[:, t] .= muladd(KdS_t, S_t \ δt, μ_t_tmin1[:, t])
        mul!(Σ_t_t[:, :, t], muladd(-KdS_t, S_t \ C, I), Σ_t_tmin1[:, :, t])
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
        mul!(G_t, Σ_t_t[:, :, t], A' / Σ_t_tmin1[:, :, t+1])

        μ_t_T[:,    t] .= muladd(G_t, μ_t_T[:, t+1] - μ_t_tmin1[:, t+1], μ_t_t[:, t])
        Σ_t_T[:, :, t] .= muladd(G_t * (Σ_t_T[:, :, t+1] - Σ_t_tmin1[:, :, t+1]), G_t', Σ_t_t[:, :, t])
    end

    G_0 = Σ0 * A' / Σ_t_tmin1[:, :, 1]
    μ_0_T = μ0 + G_0 * (μ_t_T[:, 1] - μ_t_tmin1[:, 1])
    Σ_0_T = Σ0 + G_0 * (Σ_t_T[:, :, 1] - Σ_t_tmin1[:, :, 1]) * G_0'

    if EM
        return (; μ_t_T, Σ_t_T, μ_0_T, Σ_0_T, μ_t_t, Σ_t_t, μ_t_tmin1, Σ_t_tmin1, init)
    else
        return (; μ_t_T, Σ_t_T, μ_0_T, Σ_0_T)
    end
end
