using HDF5
using Dates
using LinearAlgebra
using Plots

function run()
    h5open("./data/FORESEE_UTC_20200301_000015.hdf5", "r") do f

        data = f["raw"][]
        @show typeof(data)
        @show size(data)

        @show data[1]

        timestamp = f["timestamp"][]
        @show typeof(timestamp)

        #print(datetime.fromtimestamp(timestamp[0]))

        p = heatmap(data, color=:viridis)
        savefig(p, "./figs/heatmap.png")

    end

    return nothing
end

