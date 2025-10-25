include("loader.jl")
using Distributed

print_status = true

function get_timing_pulses(data, channel)
    is_chan(x, c) = x.channel == c
    timing_events = filter(x -> is_chan(x, channel), data)
    return map(x -> x.time, timing_events)
end


function fix_timing_pulses(pulses)
    dt = pulses[2:end] .- pulses[1:end-1]
    avg_dt = sum(dt) / length(dt)
    long_deltas = dt[dt .> avg_dt]
    long_deltas = sort(long_deltas)[1:Int64(round(length(long_deltas) * 0.7))]
    correct_long = round(sum(long_deltas) / length(long_deltas))

    short_deltas = dt[dt .<= avg_dt]
    correct_short = round(sum(short_deltas) / length(short_deltas))

    i = 1
    next_long = true
    while i < length(pulses)
        delta = abs(pulses[i] - pulses[i + 1])
        if next_long & (delta > 1.75 * correct_long)
            insert!(pulses, i + 1, pulses[i] + correct_long)
            println("inserted missing timing pulse!")
        end    
        if !next_long & (delta > 1.75 * correct_short)
            insert!(pulses, i + 1, pulses[i] + correct_short)
            println("inserted missing timing pulse!")
        end
        next_long = ! next_long
        i += 1
    end

    return pulses
end


function pairs(list)
    llist = reverse(list)
    res = []
    while length(llist) >= 2
        pair = (pop!(llist), pop!(llist))
        res = push!(res, pair)
    end
    return res
end


function remove_timing_overflows(data)
    time_const = 6.25e-8
    t_of_const = 2^30
    t_corrections = zeros(length(data))
    for i in 1:length(data) - 1
        if data[i].time > data[i + 1].time
            println("found timestamp overflow")
            t_corrections[i:end] .+= t_of_const
        end
    end

    inc_time(e, t) = event(e.time + t, e.channel, e.y)
    return map(inc_time, data, t_corrections)
end


function filter_neutron_hits(events)
    cutoff = 0.4
    y_n(e) = e.y > cutoff
    return filter(y_n, events)
end


struct LineParams
    start_time::Float64
    end_time::Float64
    i::Int64
    fwd::Bool
end


function bin_hits(hits, timing_pulses, line_count, delay_correction = 0)
    times = map(x -> x.time, hits)
    grid = zeros(line_count, line_count)
    i = 1
    fwd = true
    delay_correction *= times[2] - times[1]
    println("timing pulse delay is set to: ", delay_correction)
    for (line_start, line_end) in pairs(timing_pulses)
        print_status ? println("line ", i, "/", line_count) : 0
        line_start += delay_correction
        line_end -= delay_correction
        time_edges = LinRange(line_start, line_end, line_count)
        time_edges = fwd ? time_edges : reverse(time_edges)
        for j in 1:(line_count - 1)
            a = time_edges[fwd ? j : j + 1]
            b = time_edges[fwd ? j + 1 : j]
            grid[i, j] = sum(a .<= times .<= b)
        end
        i += 1
        fwd = !fwd
    end

    return grid
end

# function bin_hits(hits, timing_pulses, line_count, delay_correction = 0)
#     times = map(x -> x.time, hits)
#     grid = zeros(line_count, line_count)
#     i = 1
#     fwd = true
#     delay_correction *= times[2] - times[1]
#     println("timing pulse delay is set to: ", delay_correction)
#     all_params = fill(LineParams(0, 0, 0, true), line_count, line_count)
#     for (line_start, line_end) in pairs(timing_pulses)
#         line_start += delay_correction
#         line_end -= delay_correction
#         line_params = LineParams(line_start, line_end, i, fwd)
#         all_params[i] = line_params
#         println("adding params for line ", i)
#         i += 1
#         fwd = !fwd
#     end


#     function cal_line(params)
#         println("calculating line ", params.i)
#         res = zeros(line_count)
#         time_edges = LinRange(params.start_time, params.end_time, line_count)
#         time_edges = params.fwd ? time_edges : reverse(time_edges)
#         for j in 1:(line_count - 1)
#             a = time_edges[fwd ? j : j + 1]
#             b = time_edges[fwd ? j + 1 : j]
#             res[j] = sum(a .< times .< b)
#         end
#         return res
#     end

#     lines = fill([], line_count)

#     Threads.@threads for params in all_params
#         lines[params.i] = cal_line(params)
#     end
#     return transpose(hcat(lines...))
# end


function preprocess_measurements(data, timing_channel)
    println("-------- preprocessing data --------")

    println("removing timestamp overflows...")
    data = remove_timing_overflows(data)

    println("extracting timing pulses...")
    timing_pulses = get_timing_pulses(data, timing_channel)
    println("before fixing there are ", length(timing_pulses))

    println("fixing missing timestamps...")
    timing_pulses = fix_timing_pulses(timing_pulses)
    println(length(timing_pulses), " timing pulses found")

    println("running n-gamma-discrimination...")
    neutron_data = filter_neutron_hits(data)

    println("binning hits...")
    line_count = Int64(length(timing_pulses) / 2)
    return bin_hits(neutron_data, timing_pulses, line_count, 500)
end


function gen_square_scint(rows, cols, scint_size, area_size)
    print_status ? println("------- generating square scintillator -------") : 0
    scint_grid = zeros(rows, cols)
    cell_size = round(area_size / scint_size)
    a = Int64(cell_size)
    b = Int64(rows)
    c = Int64(cols)
    scint_grid[1:a, 1:a] .= 1.0
    scint_grid[1:a, (b-a):b] .= 1.0
    scint_grid[(c-a):c, 1:a] .= 1.0
    scint_grid[(c-a):c, (b-a):b] .= 1.0

    return scint_grid
end
