include("loader.jl")
include("prep_data.jl")


function run_deconv(measured, scint, iterations, print_params = false)
    beam = deconvolution(measured, scint, regularizer=Tikhonov(), iterations=iterations)[1]
    if print_params
        total_diff = sum((raw_beam .- (conv(beam, scint) ./ maximum(conv(beam, scint)))) .^ 2)
        println("total_diff = ", total_diff)
    end

    return beam
end


function run_ppl(data_file)
    measured_grid = conv_to_grid()
end


function plot_ppl(data_file)
    return 0.
end
