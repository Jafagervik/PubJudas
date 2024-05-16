@kwdef mutable struct AEArgs
    M_input_dim::Int = 75_000
    N_input_dim::Int = 2137
    hidden_dim::Int = 200000
    latent_dim::Int = 10_000
    λ::Float32 = 1.0f-3
    μ::Float32 = 1.0f-3
    epochs::Int = 500
    batch_size::Int = 16
    seed::Int = 42069
    verbose::Bool = false
    device::Function = gpu
    load::Bool = false
    name::String = "ae"
end
