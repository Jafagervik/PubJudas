mutable struct AE
    net
end

@layer AE

function AE(inp_dim::Int, hidden_dim::Int, latent_dim::Int; act=leakyrelu)
    enc1 = Dense(inp_dim, hidden_dim, act)
    enc2 = Dense(hidden_dim, latent_dim, act)

    dec1 = Dense(latent_dim, hidden_dim, act)
    dec2 = Dense(hidden_dim, inp_dim)

    return AE(Chain(enc1, enc2, dec1, dec2))
end

(ae::AE)(x) = ae.net(x)

model_loss(ae::AE, x) = Flux.mse(ae(x), flatten(x))
