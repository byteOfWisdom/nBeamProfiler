using Base: absdiff
using FFTW
using CSV
using DeconvOptim # doi:10.21105/jcon.00099
using Plots
using Images


function load_grid(fname)
    println("loading: ", fname)
    cols(a) = [view(a, :, i) for i in 1:size(a, 2)]
    data = cols(CSV.read(fname, CSV.Tables.matrix; delim=", "))

    xbins = data[1]
    ybins = data[2]
    values = data[3]

    nrows = maximum(xbins)
    ncols = maximum(ybins)

    num_grid = [[0.0 for ny in range(1, ncols + 1)] for nx in range(1, nrows + 1)]

    for i in eachindex(xbins)
        x = trunc(Int, xbins[i])
        y = trunc(Int, ybins[i])
        num_grid[x + 1][y + 1] = values[i]
    end

    return reduce(hcat, num_grid)'
end


function main()
    iterations = 1000
    iterations = parse(Int64, ARGS[3])

    raw_beam = load_grid(ARGS[1])
    scint = load_grid(ARGS[2])

    beam = deconvolution(raw_beam, scint, regularizer=Tikhonov(), iterations=iterations)[1]

    total_diff = sum((raw_beam .- (conv(beam, scint) ./ maximum(conv(beam, scint)))) .^ 2)
    println("total_diff = ", total_diff)

    norm_factor = sum(raw_beam) / sum(conv(beam, scint))
    reconvoluted = conv(beam, scint) .* norm_factor

    h1 = heatmap(raw_beam, title="measured")
    h2 = heatmap(raw_beam .- reconvoluted, title="difference")
    h3 = heatmap(beam, title="calculated beam")
    h4 = heatmap(reconvoluted, title="reconvoluted")
    # h4 = heatmap(scint, title="scintillator")
    plot(h1, h2, h3, h4, layout=(2, 2))


    gui()
end


main()
