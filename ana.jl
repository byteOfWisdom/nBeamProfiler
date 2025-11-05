#!julia
using DeconvOptim

include("loader.jl")
include("prep_data.jl")


function run_deconv(measured, scint, iterations, print_params = false)
    beam = deconvolution(measured, scint, regularizer=Tikhonov(), iterations=iterations)[1]
    if print_params
        total_diff = sum((measured .- (conv(beam, scint) ./ maximum(conv(beam, scint)))) .^ 2)
        println("total_diff = ", total_diff)
    end

    return beam
end


function run_ppl(data_file, timing_channel)
    measurements = load_file(data_file)
    prepped, lines = preprocess_measurements(measurements, timing_channel)
    scint = gen_square_scint(lines, lines, 2.54, 30.)
    print_status = true
    print_status ? println("------- running deconvolution -------") : 0
    return run_deconv(prepped, scint, 2000, true), prepped
end


function plot_ppl(data_file)
    return 0.
end


function write_csv(data, fname)
    rows, cols = size(data)
    result = []
    for i in 1:rows
        for j in 1:cols
            # result = string(i) + ", " + string(j) + ", " + string(data[i, j]) + "\n"
            result = push!(result, string(i - 1))
            result = push!(result, ", ")
            result = push!(result, string(j - 1))
            result = push!(result, ", ")
            result = push!(result, string(data[i, j]))
            result = push!(result, "\n")
        end
    end
    # println(join(result))
    write(fname, join(result))
end


function main()
    if length(ARGS) < 3
        println("too few args given")
        return
    end
    input_file = ARGS[1]
    timing_channel = parse(Int8, ARGS[2])
    output_file = ARGS[3]
    result, raw = run_ppl(input_file, timing_channel)
    println("writing to: ", output_file)
    write_csv(result, output_file)
    return result, raw
end

# res, raw = main()
