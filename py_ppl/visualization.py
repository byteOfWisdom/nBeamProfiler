from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import curve_fit
from sigfig import round         #import a easy way of scientific rounding


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

#Plots for Preview 1
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

#Plots for Preview 2
def plot_b(data, result, reconvolved_norm, diff1, diff2, args):
    # some variables for automating stuff
    scanning_length_x = 29.9 #cm
    scanning_length_y = 30 #cm
    x_lines, y_lines = np.meshgrid(np.arange(np.max(data[0]) + 1), np.arange(np.max(data[1]) + 1)) #meshgrid with number of lines
    x_units, y_units = np.meshgrid(np.linspace(0,scanning_length_x,np.max(data[0]) + 1), np.linspace(0,scanning_length_y,np.max(data[1]) + 1)) #meshgrid with lines converted to physical distance
    
    popt, pcov = curve_fit(
                            super_gaussian_2d
                            ,(x_units, y_units)
                            ,result.ravel()
                            ,p0=[1, scanning_length_x/2, scanning_length_y/2, 1.0, 1.0, 2,0]         # initial_guess
                            ,maxfev=99999999
                            ,bounds=(
                                    [0, 5, 5, 0.1, 0.1, 1,0],  # lower bounds
                                    [1, scanning_length_x-1, scanning_length_y-1, 100, 100, 100,np.inf]    # upper bounds
                                    )
                            )
    
    # Extract fitted parameters
    A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, n_fit, offset_fit = popt
    #calculate fit parameter error
    A_error, x0_error, y0_error, sigma_x_error, sigma_y_error, n_error, offset_error = np.sqrt(np.diag(pcov))
    print("------------------------------------")
    print("Fitted parameters for Supergaussian:")
    print(f"A=({A_fit:.3f}+-{A_error:.3f}), x0=({x0_fit:.3f}+-{x0_error:.3f}), y0=({y0_fit:.3f}+-{y0_error:.3f}),\n"
    f"sigma_x=({sigma_x_fit:.3f}+-{sigma_x_error:.3f}), sigma_y=({sigma_y_fit:.3f}+-{sigma_y_error:.3f}),\n"
    f"n=({n_fit:.3f}+-{n_error:.3f}), offset=({offset_fit:.3f}+-{offset_error:.3f})")
    print("------------------------------------")
    
    # use the fitted function to calculate z-values for plotting
    Z_true = super_gaussian_2d((x_units, y_units), A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, n_fit, offset_fit).reshape(np.max(data[1]) + 1, np.max(data[1]) + 1)

    # clean data from zeros because we assume error=sqrt(value) and divide by this when doing chi2
    Z_clean = []
    h_clean = []
    for i in range(len(Z_true)):
        for j in range(len(Z_true[0])):
            if Z_true[i][j] > 0 and result[i][j]>0:
                Z_clean += [Z_true[i][j]]
                h_clean += [result[i][j]]

    Z_clean = np.asarray(Z_clean).flatten()
    h_clean = np.asarray(h_clean).flatten()
    
    #calculate reduced chi2
    chi2 = np.sum( (Z_clean - h_clean)**2 / np.sqrt(h_clean)**2 ) 
    dofs = len(Z_clean) - len(popt)
    red_chi = chi2/dofs
    print("Supergaussian chi2 is: ")
    print(red_chi)

    # set axis boundaries for nice plotting
    lower_x = x0_fit-6*sigma_x_fit  #lower x value in cm for plotting
    upper_x = x0_fit+6*sigma_x_fit  #upper x value in cm for plotting
    lower_y = y0_fit-6*sigma_y_fit  #lower x value in cm for plotting
    upper_y = y0_fit+6*sigma_y_fit  #upper x value in cm for plotting


    fig = plt.figure(figsize=plt.figaspect(0.5))
    
    #RAW DATA as contour plot
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.title.set_text("raw data")
    ax.view_init(elev=45, azim=-45, roll=0)
    ax.contour(x_lines, y_lines, matrix(data), levels=100, axlim_clip=True)
    ax.contourf(x_lines, y_lines, matrix(data), zdir='x', offset=lower_x*(np.max(data[0])+1)/scanning_length_x, levels=300, cmap='rainbow', axlim_clip=True)
    ax.contourf(x_lines, y_lines, matrix(data), zdir='y', offset=upper_y*(np.max(data[0])+1)/scanning_length_y, levels=300, cmap='rainbow', axlim_clip=True)
    ax.set_zlim(0,1.1)
    ax.set_xlim(lower_x*(np.max(data[0])+1)/scanning_length_x, upper_x*(np.max(data[0])+1)/scanning_length_x) #x-axis in lines instead of cm
    ax.set_ylim(lower_y*(np.max(data[0])+1)/scanning_length_y, upper_y*(np.max(data[0])+1)/scanning_length_y) #y-axis in lines instead of cm
    ax.set_xlabel("x / lines")
    ax.set_ylabel("y / lines")
    ax.set_zlabel("normalised intensity")

    #UNFOLDED DATA as contour plot
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.title.set_text("unfolded data")
    ax.view_init(elev=45, azim=-45, roll=0)
    ax.contour(x_units, y_units, result, levels=100, axlim_clip=True)
    ax.contourf(x_units, y_units, result, zdir='x', offset=lower_x, levels=300, cmap='rainbow', axlim_clip=True)
    ax.contourf(x_units, y_units, result, zdir='y', offset=upper_y, levels=300, cmap='rainbow', axlim_clip=True)
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
    ax.contour(x_units, y_units, reconvolved_norm, levels=300, axlim_clip=True)
    ax.contourf(x_units, y_units, reconvolved_norm, zdir='x', offset=lower_x, levels=300, cmap='rainbow', axlim_clip=True)
    ax.contourf(x_units, y_units, reconvolved_norm, zdir='y', offset=upper_y, levels=300, cmap='rainbow', axlim_clip=True)
    ax.set_zlim(0,1.1)
    ax.set_xlim(lower_x,upper_x)
    ax.set_ylim(lower_y,upper_y)
    ax.set_xlabel("x / cm")
    ax.set_ylabel("y / cm")
    ax.set_zlabel("normalised intensity")

    #FITTED 2D-ELLIPTICAL SUPERGAUSSIAN as contour plot
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.title.set_text("fitted data")
    ax.view_init(elev=45, azim=-45, roll=0)
    ax.contour(x_units, y_units, Z_true, levels=300, axlim_clip=True)
    ax.contourf(x_units, y_units, Z_true, zdir='x', offset=lower_x, levels=300, cmap='rainbow', axlim_clip=True)
    ax.contourf(x_units, y_units, Z_true, zdir='y', offset=upper_y, levels=300, cmap='rainbow', axlim_clip=True)

    # fit parameters as labels in legend
    ax.plot([],[],' ', label='$\\sigma_x=$'+round(sigma_x_fit, sigma_x_error, sep='external_brackets')+'cm' )
    ax.plot([],[],' ', label='$\\sigma_y=$'+round(sigma_y_fit, sigma_y_error, sep='external_brackets')+'cm' )
    ax.plot([],[],' ', label='$n=$'+round(n_fit, n_error, sep='external_brackets') )
    ax.plot([],[],' ', label=f'$red. \\chi^2=${red_chi:.3f}' )

    # Set legend for ax and labels
    ax.legend(loc='best',handlelength=0, handletextpad=0)
    ax.set_zlim(0,1.1)
    ax.set_xlim(lower_x,upper_x)
    ax.set_ylim(lower_y,upper_y)
    ax.set_xlabel("x / cm")
    ax.set_ylabel("y / cm")
    ax.set_zlabel("normalised intensity")

    # STOPPING CRITERIA - stupid for now, do later
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

    plt.show()
