using FFTW
using CSV
using Deconvolution
using DeconvOptim
using Plots
using Images

function beam(x, y)
    δx = 100.0
    δy = 400.0
    offset_x = 50.0
    offset_y = 50.0
    x -= offset_x
    y -= offset_y
    return exp(-(x^2 / δx) - (y^2 / δy))
end

function scint(x, y)
    size = 30.0
    x = x - 50.0
    y = y - 50.0
    #    x = x % (40 * 2.5)
    #    y = y % (40 * 2.5)
    #if x <= size && y <= size && x >= 0.0 && y >= 0.0
    if x^2 <= size^2 && y^2 <= size^2
        return 1.0
    end
    return 0.0
end

function pop_grid(f, xbins, ybins, stride)
    grid = [[f(x * stride, y * stride) for y in range(1, ybins + 1)] for x in range(1, xbins + 1)]
    return reduce(hcat, grid)'
end

function shift(mat, rows, cols)
    x, y = size(mat)
    res = fill(0.0, (x, y))
    for n in range(1, x)
        sn = n + rows
        if sn < 1
            sn += x
        elseif sn > x
            sn -= x
        end
        for m in range(1, y)
            sm = m + cols
            if sm < 1
                sm += y
            elseif sm > y
                sm -= y
            end
            res[n, m] = mat[sn, sm]
        end
    end

    return res
end

bins = 40
stride = 2.5

beam_grid = pop_grid(beam, bins, bins, stride)
scint_grid = shift(pop_grid(scint, bins, bins, stride), 19, 19)

measure_grid = conv(beam_grid, scint_grid)

#reconstructed = lucy(measure_grid, scint_grid, iterations=1000)
reconstructed = deconvolution(measure_grid, scint_grid)[1]
#reconstructed = reconstructed / maximum(reconstructed)

delta = reconstructed - beam_grid

h1 = heatmap(measure_grid, title="measured")
h2 = heatmap(scint_grid, title="scint")
h3 = heatmap(reconstructed, title="beam")
h4 = heatmap(delta, title="delta")
plot(h1, h2, h3, h4, layout=(2, 2))

gui()
