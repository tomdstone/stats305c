using Pkg
Pkg.activate("Stats305c", shared=true)

using LinearAlgebra
using StateSpaceDynamics
using Pickle
using StatsPlots
using Distributions
using MultivariateStats
# using HiddenMarkovModels
using Random

## --------- Model functions  ------------

mutable struct MD_Model{T<:AbstractFloat}
    A::AbstractMatrix{T}
    C::AbstractMatrix{T}
    H::AbstractMatrix{T}
    Kmax::Int64
    k₀::Vector{T}
    z₀::Vector{T}
    Σ₀::AbstractMatrix{T}
end

nneuron = 128
nstate = 256
Kmax = 10

trans_matrix = rand(Kmax, Kmax)
trans_matrix ./= sum(trans_matrix, dims=2)

model = MD_Model(
    randn(nstate, nstate),
    randn(nneuron, nstate),
    trans_matrix,
    Kmax,
    zeros(Kmax),
    zeros(nneuron),
    I(nneuron) |> Matrix{Float64}
)

function assemble_first_k_tensor(C, kmax; T=Float64)
    u, s, v = svd(C)

    Cs = Array{T}(undef, size(C)..., kmax)

    for k in 1:kmax
        Cs[:,:,k] .= u[:, 1:k] * diagm(s[1:k]) * v'[1:k, :]
    end

    return Cs
end


function assemble_HMM(m::MD_Model{T}, Cs = nothing) where T
    nneuron, nstate = size(m.C)

    if isnothing(Cs)
        Cs = assemble_first_k_tensor(m.C, m.Kmax; T)
    end

    obsmodels = EmissionModel[
        PoissonRegressionEmission(
            nstate, nneuron, Cs[:,:,k], true, 0.
        ) for k in 1:m.Kmax
    ]

    return HiddenMarkovModel(m.H, obsmodels, m.k₀, m.Kmax)
end





function estimate_hidden_z(m::MD_Model{T}, xs, k_ests) where T

end

function estimate_hidden_k(m::MD_Model{T}, xs, z_ests) where T

end

function update_A!(m::MD_Model{T}, xs, k_ests, z_ests) where T

end

function update_C!(m::MD_Model{T}, xs, k_ests, z_ests) where T

end

# TODO: how to generally force prior on H
# NOTE: k_ests is multiple trials of data!!!!
function update_H!(m::MD_Model{T}, xs, k_ests, z_ests) where T

    # H = zeros(T, m.Kmax, m.Kmax)
    H = ones(T, m.Kmax, m.Kmax) # enforcing a prior that every transition is possible

    for k_seq in k_ests
        for t in 2:length(k_seq)
            k_t = k_seq[t]
            k_tmin1 = k_seq[t-1]

            H[k_tmin1, k_t] += one(T)
        end
    end

    m.H .= H ./ sum(H, dims=2)

    return nothing
end
