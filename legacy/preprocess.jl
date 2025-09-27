using CSV


function load_data(filename)
    transform(a) = [view(a, :, i) for i in 1:size(a, 2)]
    cols = transform(CSV.read(ARGS[1], CSV.Tables.matrix; delim=" "))

end


function classify_neutrons(long_arr, short_arr)
    neutron_cutoff = 0.4 # TODO: actually find the correct value here.
    y = (long_arr .- short_arr) ./ long_arr
    return long_arr .> short_arr .& 1
end
