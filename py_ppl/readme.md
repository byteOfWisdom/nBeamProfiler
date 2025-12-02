    input file: "data" required
    output file: "out" default value: ""
    number of line lines: "lines" default value 160
    timing_channel: "timing_chan" default value 3
    data_channel: "data_chan" default value 1
    iterations int(get_assign("iter", 1000))
    format "mesy_format", False)
    size float("size", 30.))
    scint_size float("scint_size", 2.54))
    preview int("preview", 0))
    n_gamma_cut float("n_gamma_cut", 0.))
    dt_timing float("dt_timing", 0.0))
    no_deconv bool("no_deconv", False))
    return res
