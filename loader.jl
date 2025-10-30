using CSV

print_status = true

struct event
    time::Int64
    channel::Int8
    y::Float64
end


function subset(data, condition)
    return data[condition]
end


function line_to_event(line)
    nums = split(line, ",")
    # long short time channel
    long = parse(Float64, nums[1])
    short = parse(Float64, nums[2])
    time = parse(Int64, nums[3])
    channel = parse(Int8, nums[4])
    y = (long - short) / long
    return event(time, channel, y)
end


function load_dataset(fname)
    data = map(line_to_event, readlines(fname))
    return data
end


struct ParseRes
    content::Array{Array{Float64}}
end

using Distributed

function de_mesyfy(fname)
    res = []
    conv_const = 1 / 6.25e-8 
    println("beginning load file...")
    # file = open(fname)
    println("beginning read")
    lines = readlines(fname)[2:end] # eachline(file)
    # iterate(lines) # discardes first line
    # for line in lines
    #     chunks = split(line, ",")
    #     for i in 0:15
    #         time = Int64(round(parse(Float64, chunks[1]) * conv_const))
    #         long = chunks[i + 2]
    #         short = chunks[i + 18]
    #         if long != ""
    #             res = push!(res, join([long, short, time, i], ", "))
    #         end                
    #     end
    # end
    
    function parse_mesy_line(line)
        res = []
        chunks = split(line, ",")
        for i in 0:15
            time = Int64(round(parse(Float64, chunks[1]) * conv_const))
            long = chunks[i + 2]
            short = chunks[i + 18]
            if long != ""
                res = push!(res, join([long, short, time, i], ", "))
            end                
        end
        return ParseRes(res)
    end

    res = fill(ParseRes, length(lines))
    println("beginning parse")

    @distributed for j in eachindex(lines)
        res[j] = parse_mesy_line(lines[j])
    end

    res = Iterators.flatten(res)
    return res
end


function load_mesyset(fname)
    conv_lines = de_mesyfy(fname)
    data = map(line_to_event, conv_lines)
    return data
end


function load_file(fname)
    print_status ? println("-------- loading file --------") : 0
    if length(split(readlines(fname)[1], ",")) > 6
        print_status ? println("format is mesytec software") : 0
        return load_mesyset(fname)
    end
    print_status ? println("format is daq program") : 0
    return load_dataset(fname)
end


struct EventList
    filled::Int64
    len::Int64
    events::Ptr{event}
end


function c_load_file(fname; is_mesy=false)
    line_count = countlines(fname)
    temp = @ccall "loader.so".load_file(fname::Cstring, line_count::Int64, is_mesy::Bool)::EventList
    return unsafe_wrap(Array{event}, temp.events, temp.filled; own=true)
end


function test()
    fname = "test.c"
    l = countlines(fname)
    res = @ccall "loader.so".test(fname::Cstring, l::Int64)::Ptr{event}
    arr = unsafe_wrap(Array, res, l)
    println(arr)
end

function conv_to_grid(buffer, nrows, ncols)
    return 0.
end
