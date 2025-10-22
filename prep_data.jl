include("loader.jl")

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
            println("found timestmp overflow")
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


function bin_hits(hits, timing_pulses, line_count)
    times = map(x -> x.time, hits)
    grid = zeros(line_count, line_count)
    i = 1
    fwd = true
    for (line_start, line_end) in pairs(timing_pulses)
        time_edges = LinRange(line_start, line_end, line_count)
        time_edges = fwd ? time_edges : reverse(time_edges)
        for j in 1:(line_count - 1)
            a = time_edges[j]
            b = time_edges[j + 1]
            grid[i, j] = sum(a .< times .< b)
        end
        i += 1
        fwd = !fwd
    end

    return grid
end


function preprocess_measurements(data, timing_channel)
    println("-------- preprocessing data --------")

    println("removing timestamp overflows...")
    data = remove_timing_overflows(data)

    println("extracting timing pulses...")
    timing_pulses = get_timing_pulses(data, timing_channel)

    println("fixing missing timestamps...")
    timing_pulses = fix_timing_pulses(timing_pulses)

    println("running n-gamma-discrimination...")
    neutron_data = filter_neutron_hits(data)

    println("binning hits...")
    line_count = Int64(length(timing_pulses) / 2)
    return bin_hits(neutron_data, timing_pulses, line_count)
end
