using HDF5

file_path = "../../DAS/Test/20200301_000015.hdf5"

raw_data, timestamp_data = h5open(file_path, "r") do f
    raw_data = read(f["raw"])
    timestamp_data = read(f["timestamp"])
    @show size(raw_data)
    @show length(timestamp_data)
    return raw_data, timestamp_data
end

