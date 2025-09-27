from sys import argv
import math

def csv_print(*cols):
    for i in range(len(cols[0])):
        print(", ".join([str(c[i]) for c in cols]))
        


if __name__ == "__main__":
    _ = print("args: [bin count] [size] [scan area size]") if len(argv) < 3 else None
    size = float(argv[2]) / float(argv[3])
    bin_count = int(argv[1])

    grid = [[0.0 for _ in range(bin_count)] for _ in range(bin_count)]

    high_bins = size * bin_count 

    low_cut = int(math.ceil(high_bins / 2))
    high_cut = bin_count - low_cut

    xs = []
    ys = []
    amps = []
    for i in range(bin_count):
        for j in range(bin_count):
            xs.append(i)
            ys.append(j)
            amps.append(1.0 if (i < low_cut or i > high_cut) and (j < low_cut or j > high_cut) else 0.0)

    csv_print(xs, ys, amps)
