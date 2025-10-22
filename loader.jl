using CSV

struct event
    time::Float64
    channel::Int8
    y::Float64
end


function subset(data, condition)
    return data[condition]
end


function line_to_event(line)
    nums = map(x -> parse(Float64, x), split(line, ","))
    # long short time channel
    long = nums[1]
    short = nums[2]
    time = nums[3]
    channel = Int8(nums[4])
    y = (long - short) / long
    return event(time, channel, y)
end


function load_dataset(fname)
    data = map(line_to_event, readlines(fname))
    return data
end


function de_mesyfy(fname)
    res = []
    conv_const = 1 / 6.25e-8
    lines = readlines(fname)
    for line in lines[2:end]
        chunks = split(line, ",")
        for i in 0:15
            time = parse(Float64, chunks[1]) * conv_const
            long = chunks[i + 2]
            short = chunks[i + 18]
            if long != ""
                res = push!(res, join([long, short, time, i], ", "))
            end                
        end
    end
    return res
end


function load_mesyset(fname)
    conv_lines = de_mesyfy(fname)
    data = map(line_to_event, conv_lines)
    return data
end


function load_file(fname)
    if length(split(readlines(fname)[1], ",")) > 6
        return load_mesyset(fname)
    end
    return load_dataset(fname)
end


function conv_to_grid(buffer, nrows, ncols)
    return 0.
end
