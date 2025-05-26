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

    for x in range(1, nrows + 1)
        for y in range(1, nrows + 1)
            if length(grid[x][y]) == 0
                grid[x][y] = [0.0]
            end
            num_grid[x][y] = sum(grid[x][y]) / length(grid[x][y])
        end
    end

    num_grid = reduce(hcat, num_grid)'
    return num_grid
end


function scint_func(x, y)
    size = 40.0
    x = x - 50.0
    y = y - 50.0
    #    x = x % (40 * 2.5)
    #    y = y % (40 * 2.5)
    #if x <= size && y <= size && x >= 0.0 && y >= 0.0
    if x^2 < size^2 && y^2 < size^2
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

function pop_grid(f, xbins, ybins, stride_x, stride_y)
    grid = [[f(x * stride_x, y * stride_y) for y in range(1, ybins + 1)] for x in range(1, xbins + 1)]
    return reduce(hcat, grid)'
end

num_grid = even_grid(fluency)

scint = shift(pop_grid(scint_func, nrows, nrows, xstep, ystep), convert(Int, nrows / 2) - 2, convert(Int, nrows / 2) - 2)

beam = lucy(num_grid, scint, iterations=10)

h1 = heatmap(num_grid, title="measured")
h2 = heatmap(scint, title="scintilator")
h3 = heatmap(beam, title="cal beam")
plot(h1, h2, h3, layout=(2, 2))
gui()
