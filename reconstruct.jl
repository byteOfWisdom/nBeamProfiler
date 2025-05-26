using FFTW
using CSV
using Deconvolution
using DeconvOptim
using Plots
using Images


cols(a) = [view(a, :, i) for i in 1:size(a, 2)]
data = cols(CSV.read(ARGS[1], CSV.Tables.matrix; delim=" "))

times = data[1]
fluency = data[2]
x_list = data[3]
y_list = data[4]

nrows = 40
xsize = 100.0
ysize = 100.0
ystep = ysize / nrows
xstep = xsize / nrows

row_y(y) = convert(Int64, round(abs(y) / ystep)) + 1
col_x(x) = convert(Int64, round(abs(x) / xstep)) + 1

function scint_psf(x, y)
    δx = 100.0
    δy = 100.0
    x -= xsize / 2.0
    y -= ysize / 2.0
    return exp(-(x^2 / δx) - (y^2 / δy))
end


function even_grid(samples)
    # bunch data into even chunks
    grid = [[[] for ny in range(1, nrows + 1)] for nx in range(1, nrows + 1)]

    for i in range(1, length(samples))
        x = col_x(x_list[i])
        y = row_y(y_list[i])
        append!(grid[x][y], samples[i])
    end

    #println(grid)

    num_grid = [[0.0 for ny in range(1, nrows + 1)] for nx in range(1, nrows + 1)]

    for x in range(1, nrows)
        for y in range(1, nrows)
            if length(grid[x][y]) == 0
                grid[x][y] = [0.0]
            end
            num_grid[x][y] = sum(grid[x][y]) / length(grid[x][y])
        end
    end

    num_grid = reduce(hcat, num_grid)'
    return num_grid
end


num_grid = even_grid(fluency)

scint_size = 40.0
scint = fill(0.0, (nrows + 1, nrows + 1))
sx = col_x(scint_size / 4)
sy = row_y(scint_size / 4)

scint[1:sx, 1:sy] .= 1.0
scint[nrows-sx+1:nrows+1, 1:sy] .= 1.0
scint[1:sx, nrows+1-sy:nrows+1] .= 1.0
scint[nrows-sx+1:nrows+1, nrows+1-sy:nrows+1] .= 1.0

beam = lucy(num_grid, scint, iterations=1000)

#heatmap(temp_scint)
heatmap(scint)
#heatmap(num_grid)
gui()
