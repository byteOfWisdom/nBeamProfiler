from sys import argv
import fileinput


def main():
    in_file, out_file = argv[1], argv[2]

    out_handle = open(out_file, "w")
    in_handle = fileinput.input(in_file)

    _ = next(in_handle) # discard header line

    timeconst = 1 / 6.25e-8

    for line in in_handle:
        cols = line.split(",")
        time = int(float(cols[0]) * timeconst)
        for channel in range(16):
            long = cols[channel + 1]
            short = cols[channel + 17]
            if long != '':
                # short, long, timestamp, channel, 0
                out_handle.write(f"{long}, {short}, {time}, {channel}, 0\n")

            channel += 1

            
        


if __name__ == "__main__":
    main()
