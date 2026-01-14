#!python3
from sys import argv
import deconv
import data_loading
import square_scint
import post_process
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.signal import convolve2d as conv2
from scipy.optimize import curve_fit


def to_csv(mat):
    res = ""
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            res += f"{i}, {j}, {mat[i][j]}\n"
    return res


def get_assign(tag, default=None):
    for arg in argv:
        if arg.startswith(tag + "="):
            return "".join(arg.split("=")[1:])
    return default


def args_to_dict():
    res = {}
    res["in_file"] = get_assign("data")
    res["out_file"] = get_assign("out", "")
    res["lines"] = int(get_assign("lines", 160))
    res["timing_channel"] = int(get_assign("timing_chan", 3))
    res["data_channel"] = int(get_assign("data_chan", 2))
    res["iterations"] = int(get_assign("iter", 1000))
    res["format"] = get_assign("mesy_format", False)
    res["size"] = float(get_assign("size", 30.))
    res["scint_size"] = float(get_assign("scint_size", 2.54))
    res["preview"] = int(get_assign("preview", 0))
    res["n_gamma_cut"] = float(get_assign("n_gamma_cut", 0.))
    res["dt_timing"] = float(get_assign("dt_timing", 0.0))
    res["no_deconv"] = bool(get_assign("no_deconv", False))
    res["scint_amp_mod"] = float(get_assign("scint_amp_mod", 0.25))
    res["fit_beam_shape"] = bool(get_assign("fit_beam", False))
    return res


def missing_args(args):
    for a in args:
        if isinstance(get_assign(a), type(None)):
            return True
    return False


def matrix(arr):
    xdim = max(arr[0])
    ydim = max(arr[1])
    res = np.zeros((xdim + 1, ydim + 1))
    for x, y, v in zip(arr[0], arr[1], arr[2]):
        res[x][y] = v
    return res

# Define the 2D super-Gaussian model
def super_gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, n):
    x, y = coords
    # Ensure sigma values and exponent are positive
    sigma_x = abs(sigma_x)
    sigma_y = abs(sigma_y)
    n = abs(n)
    exponent = ((x - x0) / sigma_x) ** (2 * n) + ((y - y0) / sigma_y) ** (2 * n)
    return (A * np.exp(-exponent) ).ravel()




def main():
    if missing_args(["data"]):
        print("missing relevant arguments")
        return
    args = args_to_dict()
    data_file = args['in_file']
    # out_file = args['out_file']
    mesy_format = args['format']
    print("---- loading input file ----")
    data = data_loading.load_file(data_file,
                                  mesy_format,
                                  args['lines'],
                                  args['timing_channel'],
                                  args['data_channel'],
                                  args['n_gamma_cut'],
                                  args['dt_timing'])
    
    long_data , short_data = data_loading.PSD_Heatmap(data_file)

    if args['no_deconv']:
        plt.imshow(matrix(data))
        plt.show()
        return

    #DECONVOLUTION
    print("---- generating scintillator mask ----")
    scint = square_scint.square_scint(args["lines"], args["scint_size"], args['size'], args['scint_amp_mod'])
    print("---- running deconvolution ----")
    result, info = deconv.deconv_rl(matrix(data), scint, args["iterations"])
    print(info)

    #RE-CONVOLUTION
    reconvolved = np.array(conv2(result, scint, mode='same'))
    reconvolved_norm = np.array(reconvolved) / np.amax(reconvolved)
    print("---- Re-convolution successful ----")

    # ITERATIVE RE-CONVOLUTING and calculating chi2/stopping criteria for every iteration 
    diff1 = []
    diff2 = []
    result_old =[]
    result_next =[]
    reconvolved_old = []
    reconvolved_next = []
    for i in range(args["iterations"]+1):
        result_next, info2 = deconv.deconv_rl(matrix(data), scint, i)
        reconvolved_next = np.array(conv2(result_next, scint, mode='same'))
        reconvolved_next = np.array(reconvolved_next) / np.amax(reconvolved_next)
        if i > 0:
            # print( np.sum( (result_next - result_old)**2 ) )
            diff1 = np.append(diff1, np.sqrt(np.sum( (result_next - result_old)**2) ))          #difference between iteration
            diff2 = np.append(diff2, np.sqrt( np.sum( ((reconvolved_old - matrix(data))**2)    ))  )    #differnce between n-th iteration and raw Data
        result_old = result_next
        reconvolved_old = reconvolved_next
        # print(diff1)
        # print(diff2)
        # print(info2 + " for " + str(i) + " times")
    min_index = np.argmin(diff2)
    print("Minimum reached after " + str(min_index+1) + " iterations with " + str(diff2[min_index]))


    csv_data = to_csv(result)
    if args['out_file'] != "":
        with open(args['out_file'], "w") as handle:
            handle.write(csv_data)
    # print(csv_data)

    if args['fit_beam_shape']:
        print("---- attempting to fit a shape to the data ----")
        post_process.fit_beam(result)

    if args['preview'] == 1 or args['preview'] == 3:    #2D-Plots:
        fig, ax = plt.subplots(2, 2)
        #
        ax[0, 0].title.set_text("raw data")
        ax[0, 0].imshow(matrix(data))
        #
        ax[0, 1].title.set_text("scint function")
        ax[0, 1].imshow(scint)
        #
        ax[1, 0].title.set_text("PSD")
        ax[1, 0].hist2d(long_data, ((long_data - short_data) / (long_data)), bins=500, cmap='rainbow', norm=matplotlib.colors.LogNorm())
        ax[1, 0].axhline(y=args['n_gamma_cut'], color='black', linestyle='-')
        ax[1, 0].text(60000, args['n_gamma_cut']+0.02, 'neutrons', fontsize=12, color='black', ha='center', va='center')
        ax[1, 0].text(60000, args['n_gamma_cut']-0.02, 'gammas', fontsize=12, color='black', ha='center', va='center')
        ax[1, 0].set_xlabel("long")
        ax[1, 0].set_ylabel("(long-short)/long")
        #
        ax[1, 1].title.set_text("unfolded data")
        ax[1, 1].imshow(result)
        #
        fig.tight_layout()
        plt.show()

    if args['preview'] == 2 or args['preview'] == 3:        #3D-Plots
        lower_x = 10    #lower x value in cm for plotting
        upper_x = 22    #upper x value in cm for plotting
        lower_y = 10    #lower y value in cm for plotting
        upper_y = 22    #upper y value in cm for plotting
        fig = plt.figure(figsize=plt.figaspect(0.5))
        
        #RAW DATA as contour plot
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.title.set_text("raw data")
        ax.view_init(elev=45, azim=-45, roll=0)
        x, y = np.meshgrid(np.arange(np.max(data[0]) + 1), np.arange(np.max(data[1]) + 1))            #meshgrid with number of lines
        ax.contour(x, y, matrix(data), levels=100, axlim_clip=True)
        ax.contourf(x, y, matrix(data), zdir='x', offset=lower_x*(np.max(data[0])+1)/30, levels=300, cmap='rainbow', axlim_clip=True)
        ax.contourf(x, y, matrix(data), zdir='y', offset=upper_y*(np.max(data[0])+1)/30, levels=300, cmap='rainbow', axlim_clip=True)
        ax.set_zlim(0,1.1)
        ax.set_xlim(lower_x*(np.max(data[0])+1)/30, upper_x*(np.max(data[0])+1)/30) #x-axis in lines instead of cm
        ax.set_ylim(lower_y*(np.max(data[0])+1)/30, upper_y*(np.max(data[0])+1)/30) #y-axis in lines instead of cm
        ax.set_xlabel("x / lines")
        ax.set_ylabel("y / lines")
        ax.set_zlabel("normalised intensity")

        #UNFOLDED DATA as contour plot
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.title.set_text("unfolded data")
        ax.view_init(elev=45, azim=-45, roll=0)
        x, y = np.meshgrid(np.linspace(0,30,np.max(data[0]) + 1), np.linspace(0,30,np.max(data[1]) + 1))#meshgrid with lines to 30cm
        # ax.contour(x, y, result, levels=100, axlim_clip=True)
        # ax.contourf(x, y, result, zdir='x', offset=lower_x, levels=300, cmap='rainbow', axlim_clip=True)
        # ax.contourf(x, y, result, zdir='y', offset=upper_y, levels=300, cmap='rainbow', axlim_clip=True)               #projection with scaled to 30cm

        #Test Data for 2D Super Gaus
        true_params = [1, 16, 16, 2.0, 2.0, 2]  # A, x0, y0, sigma_x, sigma_y, n, offset
        Z_true = super_gaussian_2d((x, y), *true_params).reshape(np.max(data[1]) + 1, np.max(data[1]) + 1)
        ax.contour(x, y, Z_true, levels=300, axlim_clip=True)
        ax.contourf(x, y, Z_true, zdir='x', offset=lower_x, levels=300, cmap='rainbow', axlim_clip=True)
        ax.contourf(x, y, Z_true, zdir='y', offset=upper_y, levels=300, cmap='rainbow', axlim_clip=True)



        # popt, pcov = curve_fit(
        #                         super_gaussian_2d
        #                         ,(x, y)
        #                         ,result.ravel()
        #                         ,p0=[1, 16, 16, 5, 5, 1]         # initial_guess
        #                         ,maxfev=99999999
        #                         ,bounds=(
        #                                 [0, 0, 0, 0.1, 0.1, 1],  # lower bounds
        #                                 [1, 30, 30, 500, 500, 10]    # upper bounds
        #                                 )
        #                         )
        # # Extract fitted parameters
        # A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, n_fit, offset_fit = popt
        # print("Fitted parameters:")
        # print(f"A={A_fit:.3f}, x0={x0_fit:.3f}, y0={y0_fit:.3f}, "
        # f"sigma_x={sigma_x_fit:.3f}, sigma_y={sigma_y_fit:.3f}, "
        # f"n={n_fit:.3f}, offset={offset_fit:.3f}")

        ax.set_zlim(0,1.1)
        ax.set_xlim(lower_x,upper_x)
        ax.set_ylim(lower_y,upper_y)
        ax.set_xlabel("x / cm")
        ax.set_ylabel("y / cm")
        ax.set_zlabel("normalised intensity")
       

        #RE-FOLDED DATA as contour plot
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.title.set_text("refolded data")
        ax.view_init(elev=45, azim=-45, roll=0)
        x, y = np.meshgrid(np.linspace(0,30,np.max(data[0]) + 1), np.linspace(0,30,np.max(data[1]) + 1))#meshgrid with lines to 30cm
        ax.contour(x, y, reconvolved_norm, levels=300, axlim_clip=True)
        ax.contourf(x, y, reconvolved_norm, zdir='x', offset=lower_x, levels=300, cmap='rainbow', axlim_clip=True)
        ax.contourf(x, y, reconvolved_norm, zdir='y', offset=upper_y, levels=300, cmap='rainbow', axlim_clip=True)               #projection with scaled to 30cm
        ax.set_zlim(0,1.1)
        ax.set_xlim(lower_x,upper_x)
        ax.set_ylim(lower_y,upper_y)
        ax.set_xlabel("x / cm")
        ax.set_ylabel("y / cm")
        ax.set_zlabel("normalised intensity")

        #STOPPING CRITERIA
        ax = fig.add_subplot(2, 2, 4)
        ax.title.set_text("change in unfolding")
        x = np.linspace(1,args["iterations"], num=args["iterations"])
        ax.plot(x,diff1, color='r', label='change between iterations')
        ax.plot(x,diff2, color='blue', label='diff between re-convoluted and raw data')
        # ax.set_xlim(0,30)
        # ax.set_ylim(0,30)
        # ax.set_xlim(lower_x,upper_x)
        # ax.set_ylim(lower_y,upper_y)
        ax.grid()
        ax.set_yscale('log')
        ax.set_xlabel("no. of Richardson Lucy iterations")
        ax.set_ylabel("sqrt(sum( (iter_n+1 - iter_n)**2 ) )")
        ax.legend(loc="best")

        plt.show()
        # print(np.max(data[0]))
        # print(np.max(data[1]))
    # data_loading.csv_print(out_file, resul)


if __name__ == "__main__":
    main()
