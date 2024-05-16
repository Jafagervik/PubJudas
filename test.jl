using HDF5
using Dates
using LinearAlgebra
using Plots

# 75 000 , 2137
function run()
    h5open("./data/FORESEE_UTC_20200301_000015.hdf5", "r") do f

        data = f["raw"][]
        @show typeof(data)
        @show size(data)

        timestamp = f["timestamp"][]

        #print(datetime.fromtimestamp(timestamp[0]))

        p = heatmap(1:size(data, 1),
            1:size(data, 2), data,
            c=:viridis,
            xlabel="Channels", ylabel="Time",
            title="YEET")
        savefig(p, "./figs/heatmap.png")

    end

    return nothing
end

