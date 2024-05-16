using Flux
using CUDA
using Random
using JLD2
using Plots

include("hyperparams.jl")
include("model.jl")
include("utils.jl")


function run(data, args)
     seed_all!(args.seed)

    model = AE(args.M_input_dim * args.N_input_dim, args.hidden_dim, args.latent_dim)

    if args.load
        @info "Loading data.."
        ms = JLD2.load("checkpoints/best_model.jld2", "model_state")
        Flux.loadmodel!(model, ms)
    end

    model = model |> args.device

    loader, _ = get_data()

    opt = Adam(args.μ)
    θ = Flux.params(model)

    losses = Vector{Float32}(undef, args.epochs)
    loss = Inf32

    for epoch ∈ 1:args.epochs
        @info "Epoch $epoch"
        ℒ = 0.0f0
        for d in loader
            ∇ = Flux.gradient(θ) do
                loss = model_loss(model, d |> args.device)
                return loss
            end

            Flux.update!(opt, θ, ∇)

            ℒ += loss

            GC.gc()
        end

        epoch % args.verbose == 0 && println("ℒ = $ℒ")

        @inbounds losses[epoch] = ℒ
end
