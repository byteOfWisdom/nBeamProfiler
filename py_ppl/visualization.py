from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import curve_fit


def matrix(arr):
    xdim = max(arr[0])
    ydim = max(arr[1])
    res = np.zeros((xdim + 1, ydim + 1))
    for x, y, v in zip(arr[0], arr[1], arr[2]):
        res[x][y] = v
    return res


# Define the 2D super-Gaussian model
def super_gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, n, offset):
    x, y = coords
    # Ensure sigma values and exponent are positive
    sigma_x = abs(sigma_x)
    sigma_y = abs(sigma_y)
    n = abs(n)
    # exponent = -((x - x0) / sigma_x) ** (2*n) - ((y - y0) / sigma_y) ** (2*n)              #rectangular gaussian beam 
    exponent = -( ( (x - x0)/(2*sigma_x) )  **2   + ( (y - y0)/(2*sigma_y) )**2 ) ** n       #elliptical gaussian beam, thanks wikipedia :)
    return (A * np.exp(exponent) + offset).ravel()


def plot_a(data, scint, long_data, short_data, result, args):
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].title.set_text("raw data")
    ax[0, 0].imshow(matrix(data))

    ax[0, 1].title.set_text("scint function")
    ax[0, 1].imshow(scint)

    ax[1, 0].title.set_text("PSD")
    ax[1, 0].hist2d(long_data, ((long_data - short_data) / (long_data)), bins=500, cmap='rainbow', norm=matplotlib.colors.LogNorm())
    ax[1, 0].axhline(y=args['n_gamma_cut'], color='black', linestyle='-')
    ax[1, 0].text(60000, args['n_gamma_cut']+0.02, 'neutrons', fontsize=12, color='black', ha='center', va='center')
    ax[1, 0].text(60000, args['n_gamma_cut']-0.02, 'gammas', fontsize=12, color='black', ha='center', va='center')
    ax[1, 0].set_xlabel("long")
    ax[1, 0].set_ylabel("(long-short)/long")

    ax[1, 1].title.set_text("unfolded data")
    ax[1, 1].imshow(result)

    fig.tight_layout()
    plt.show()


def plot_b(data, result, reconvolved_norm, diff1, diff2, args):
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
    ax.contour(x, y, result, levels=100, axlim_clip=True)
    ax.contourf(x, y, result, zdir='x', offset=lower_x, levels=300, cmap='rainbow', axlim_clip=True)
    ax.contourf(x, y, result, zdir='y', offset=upper_y, levels=300, cmap='rainbow', axlim_clip=True)               #projection with scaled to 30cm
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

    # #STOPPING CRITERIA - stupid for now, do later
    # ax = fig.add_subplot(2, 2, 4)
    # ax.title.set_text("change in unfolding")
    # x = np.linspace(1,args["iterations"], num=args["iterations"])
    # ax.plot(x,diff1, color='r', label='change between iterations')
    # ax.plot(x,diff2, color='blue', label='diff between re-convoluted and raw data')
    # # ax.set_xlim(0,30)
    # # ax.set_ylim(0,30)
    # # ax.set_xlim(lower_x,upper_x)
    # # ax.set_ylim(lower_y,upper_y)
    # ax.grid()
    # ax.set_yscale('log')
    # ax.set_xlabel("no. of Richardson Lucy iterations")
    # ax.set_ylabel("sqrt(sum( (iter_n+1 - iter_n)**2 ) )")
    # ax.legend(loc="best")


    #FITTED 2D-ELLIPTICAL SUPERGAUSSIAN as contour plot
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.title.set_text("fitted data")
    ax.view_init(elev=45, azim=-45, roll=0)
    x, y = np.meshgrid(np.linspace(0,30,np.max(data[0]) + 1), np.linspace(0,30,np.max(data[1]) + 1))#meshgrid with lines to 30cm

    #Test Data for plotting a 2D Super Gaus
    # true_params = [1, 16, 16, 1.0, 1.0, 5]  # A, x0, y0, sigma_x, sigma_y, n
    # Z_true = super_gaussian_2d((x, y), *true_params).reshape(np.max(data[1]) + 1, np.max(data[1]) + 1)
    # ax.contour(x, y, Z_true, levels=300, axlim_clip=True)
    # ax.contourf(x, y, Z_true, zdir='x', offset=lower_x, levels=300, cmap='rainbow', axlim_clip=True)
    # ax.contourf(x, y, Z_true, zdir='y', offset=upper_y, levels=300, cmap='rainbow', axlim_clip=True)

    popt, pcov = curve_fit(
                            super_gaussian_2d
                            ,(x, y)
                            ,result.ravel()
                            ,p0=[1, 16, 16, 1.0, 1.0, 15,0]         # initial_guess
                            ,maxfev=99999999
                            ,bounds=(
                                    [0, 0, 0, 0.1, 0.1, 1,0],  # lower bounds
                                    [1, 30, 30, 500, 500, 100,np.inf]    # upper bounds
                                    )
                            )
    
    # Extract fitted parameters
    # A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, n_fit, offset_fit = popt
    A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, n_fit, offset_fit = popt
    #calculate fit parameter error
    perr = np.sqrt(np.diag(pcov))
    A_error         = perr[0] 
    x0_error        = perr[1] 
    y0_error        = perr[2] 
    sigma_x_error   = perr[3] 
    sigma_y_error   = perr[4] 
    n_error         = perr[5] 
    offset_error    = perr[6] 
    print("------------------------------------")
    print("Fitted parameters for Supergaussian:")
    print(f"A=({A_fit:.3f}+-{A_error:.3f}), x0=({x0_fit:.3f}+-{x0_error:.3f}), y0=({y0_fit:.3f}+-{y0_error:.3f}),\n"
    f"sigma_x=({sigma_x_fit:.3f}+-{sigma_x_error:.3f}), sigma_y=({sigma_y_fit:.3f}+-{sigma_y_error:.3f}),\n"
    f"n=({n_fit:.3f}+-{n_error:.3f}), offset=({offset_fit:.3f}+-{offset_error:.3f})")
    print("------------------------------------")

    Z_true = super_gaussian_2d((x, y), A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, n_fit, offset_fit).reshape(np.max(data[1]) + 1, np.max(data[1]) + 1)
    ax.contour(x, y, Z_true, levels=300, axlim_clip=True)
    ax.contourf(x, y, Z_true, zdir='x', offset=lower_x, levels=300, cmap='rainbow', axlim_clip=True)
    ax.contourf(x, y, Z_true, zdir='y', offset=upper_y, levels=300, cmap='rainbow', axlim_clip=True)

    ax.set_zlim(0,1.1)
    ax.set_xlim(lower_x,upper_x)
    ax.set_ylim(lower_y,upper_y)
    ax.set_xlabel("x / cm")
    ax.set_ylabel("y / cm")
    ax.set_zlabel("normalised intensity")

    plt.show()
