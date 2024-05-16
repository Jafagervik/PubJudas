using HDF5
using Dates
using LinearAlgebra
using Plots

# hz = 125
# 75 000 , 2137
function run()
    h5open("./data/FORESEE_UTC_20200301_000015.hdf5", "r") do f
        data = f["raw"][]
        @show typeof(data)
        @show size(data)

        timestamps = f["timestamp"][]

        @show timestamps[1]

        @show unix2datetime(timestamps[1])

        #dasmap(data)
    end

    return nothing
end

function dasmap(data, timestamps)
    p = heatmap(1:size(data, 1),
        1:size(data, 2), data,
        c=:viridis,
        xlabel="Channels", ylabel="Time", xticks=false,
        title="YEET")
    dt = Dates.format.(timestamps, "HH:MM:SS")
    plot!(p, xticks=(timestamps, dt))
    savefig(p, "./figs/heatmap.png")

    return nothing
end

