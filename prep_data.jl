include("loader.jl")

function timing_pulses(data, channel)
    is_chan(x, c) = x.channel == c
    timing_events = filter(x -> is_chan(x, channel), data)
    return map(x -> x.time, timing_events)
end


function fix_timing_pulses(pulses)
    dt = pulses[2:end] - pulses[1:end-1]
    avg_dt = sum(dt) / length(dt)
    long_deltas = filter(dt .> avg_dt, dt)
    correct_long = sum(long_deltas) / length(long_deltas)

    short_deltas = filter(dt .<= avg_dt, dt)
    correct_short = sum(short_deltas) / length(short_deltas)


    i = 1
    next_long = true
    while i < length(pulses)
        delta = abs(timing_data[i] - timing_data[i + 1])
        if next_long & (delta > 1.75 * correct_long)
            pulses = insert!(pulses, i + 1, pulses[i] + correct_long)
            println("inserted missing timing pulse!")
        end    
        if next_short & (delta > 1.75 * correct_short)
            pulses = insert!(pulses, i + 1, pulses[i] + correct_short)
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
    while length(llist > 2)
        res = push!(pop!(llist), pop!(llist))
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


function preprocess_measurements(data, timing_channel)
    println("-------- preprocessing data --------")
    println("removing timestamp overflows...")
    data = remove_timing_overflows(data)
    println("extracting timing pulses...")
    timing_pulses = timing_pulses(data, timing_channel)
    println("fixing missing timestamps...")
    timing_pulses = fix_timing_pulses(timing_pulses)
    return [0]
end
