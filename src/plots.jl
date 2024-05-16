
function dasmap(data, timestamps)
    M = 100
    N = 100
    d = ones(M, N)
    p = heatmap(1:M,
        1:N, d,
        c=:viridis,
        xlabel="Channels", ylabel="Time", xticks=false,
        title="YEET")
    dt = Dates.format.(timestamps, "HH:MM:SS")
    plot!(p, xticks=(timestamps, dt))
    savefig(p, "./figs/heatmap.png")

    return nothing
end
