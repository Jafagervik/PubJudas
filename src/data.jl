
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


