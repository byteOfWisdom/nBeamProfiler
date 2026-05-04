from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import curve_fit
from sigfig import round         #import an easy way of scientific rounding


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
    return (A * np.exp(exponent) + offset).ravel()  #with offset
    # return (A * np.exp(exponent)).ravel()  #without offset

#Plot for Preview 4
def plot_c(time_edges, neutron_hits, timing_pulses, long_data, short_data, args):
    fig, ax = plt.subplots(2, 1)

    # print(timing_pulses*6.25e-8)
    # plt.figure(figsize=(16, 8), dpi=50)
    ax[0].title.set_text("Neutron Counts during Scan")
    ax[0].plot(0.5 * (time_edges[1:] + time_edges[:-1])*6.25e-8, neutron_hits, label='neutron count')
    ax[0].vlines(x=timing_pulses[0]*6.25e-8, ymin=-500, ymax=0,color='red', linestyle='-', label='timing pulses') #sneaky hack to get the label into legend
    for xc in timing_pulses*6.25e-8:
        ax[0].vlines(x=xc, ymin=-500, ymax=0,color='red', linestyle='-')
    ax[0].set_xlabel("time") #what the duck is the unit of this axis? nano seconds? scan should have taken around 30cm/(5cm/s)*80 = 480s = 8min
    ax[0].set_ylabel("neutron count")
    ax[0].set_xlim(timing_pulses[0]*6.25e-8 -10,timing_pulses[-1]*6.25e-8 +10 ) #start plotting 10s before the first pulse and stop 10s after the last one
    ax[0].legend(loc="best")
    # plt.show()

    # PULSE SHAPE DISCRIMINATION plot
    ax[1].title.set_text("Pulse Shape Discrimination")
    ax[1].hist2d(long_data, ((long_data - short_data) / (long_data)), bins=500, cmap='rainbow', norm=matplotlib.colors.LogNorm())
    ax[1].axhline(y=args['n_gamma_cut'], color='black', linestyle='-')
    ax[1].text(60000, args['n_gamma_cut']+0.02, 'neutrons', fontsize=12, color='black', ha='center', va='center')
    ax[1].text(60000, args['n_gamma_cut']-0.02, 'gammas', fontsize=12, color='black', ha='center', va='center')
    ax[1].set_xlabel("long")
    ax[1].set_ylabel("(long-short)/long")
    fig.tight_layout()
    plt.show()

#Plots for Preview 1
def plot_a(data, scint, reconvolved_norm, result, args):
    fig, ax = plt.subplots(2, 2)

    # RAW DATA - Heatmap
    ax[0, 0].title.set_text("Raw Data - Heatmap")
    ax[0, 0].imshow(matrix(data))

    # POINT SPREAD FUNCTION of the scintillator
    ax[0, 1].title.set_text("Scintillator - Point Spread Function")
    ax[0, 1].imshow(scint)

    # PULSE SHAPE DISCRIMINATION plot
    # ax[1, 0].title.set_text("Pulse Shape Discrimination")
    # ax[1, 0].hist2d(long_data, ((long_data - short_data) / (long_data)), bins=500, cmap='rainbow', norm=matplotlib.colors.LogNorm())
    # ax[1, 0].axhline(y=args['n_gamma_cut'], color='black', linestyle='-')
    # ax[1, 0].text(60000, args['n_gamma_cut']+0.02, 'neutrons', fontsize=12, color='black', ha='center', va='center')
    # ax[1, 0].text(60000, args['n_gamma_cut']-0.02, 'gammas', fontsize=12, color='black', ha='center', va='center')
    # ax[1, 0].set_xlabel("long")
    # ax[1, 0].set_ylabel("(long-short)/long")

    # Re-CONVOLVED DATA - Heatmap
    ax[1, 0].title.set_text("Re-convolved Data - Heatmap")
    ax[1, 0].imshow(reconvolved_norm)

    # DECONVOLVED DATA - Heatmap
    ax[1, 1].title.set_text("Deconvolved Data - Heatmap")
    ax[1, 1].imshow(result)

    fig.tight_layout()
    plt.show()

#Plots for Preview 2
def plot_b(data, result, reconvolved_norm, diff1, diff2, args):
    # some variables for calculating stuff
    scanning_length_x = 30 #cm
    scanning_length_y = 30 #cm

    # create meshgrid with number of lines
    x_lines, y_lines = np.meshgrid( np.arange(np.max(data[0]) + 1),
                                    np.arange(np.max(data[1]) + 1))
    
    # create meshgrid with number of lines lines converted to physical distances
    x_units, y_units = np.meshgrid( np.linspace(0,scanning_length_x,np.max(data[0]) + 1), 
                                    np.linspace(0,scanning_length_y,np.max(data[1]) + 1)) 

    # fit supergaussian to data
    popt, pcov = curve_fit(
                            super_gaussian_2d
                            ,(x_units, y_units) # fit data to axis in physical units
                            ,result.ravel()
                            # ,sigma = np.sqrt(result.ravel())
                            # ,absolute_sigma = True
                            ,p0=[1, scanning_length_x/2, scanning_length_y/2, 1.0, 1.0, 2,0]  # initial_guess for fitting parameters
                            ,maxfev=99999999
                            ,bounds=(
                                    [0, 5, 5, 0.1, 0.1, 1,0],  # lower bounds of fitting parameters
                                    [1, scanning_length_x-1, scanning_length_y-1, 100, 100, 100,np.inf]    # upper bounds of fitting parameters
                                    )
                            )
    
    # Extract fitted parameters into variables for better reading
    A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, n_fit, offset_fit = popt
    # calculate fitted parameters-error
    A_error, x0_error, y0_error, sigma_x_error, sigma_y_error, n_error, offset_error = np.sqrt(np.diag(pcov))

    # print fitted parameters to console
    print("------------------------------------")
    print("Fitted parameters for Supergaussian:")
    print(
    'A=' + str(round(A_fit, A_error, sep='external_brackets'))+', ' + 'offset=' + str(round(offset_fit, offset_error, sep='external_brackets'))+', \n'
    'x0=' + str(round(x0_fit, x0_error, sep='external_brackets'))+'cm, ' + 'y0=' + str(round(y0_fit, y0_error, sep='external_brackets'))+'cm, \n'
    'sigma_x=' + str(round(sigma_x_fit, sigma_x_error, sep='external_brackets'))+'cm, ' + 'sigma_y=' + str(round(sigma_y_fit, sigma_y_error, sep='external_brackets'))+'cm, \n'
    'n=' + str(round(n_fit, n_error, sep='external_brackets'))
        )
    print("------------------------------------")
    # print("Latex-Line for tables:")
    # print(str(round(sigma_x_fit, sigma_x_error, sep='external_brackets')) + ' & ' + str(round(sigma_y_fit, sigma_y_error, sep='external_brackets')) + ' & ' +
    #       str(round(n_fit, n_error, sep='external_brackets')) + ' & ' + str(round(A_fit, A_error, sep='external_brackets')) + ' & ' + 
    #       str(round(x0_fit, x0_error, sep='external_brackets')) + ' & ' + str(round(y0_fit, y0_error, sep='external_brackets')) + ' & ' +
    #       str(round(offset_fit, offset_error, sep='external_brackets')) + ' \\\\'
    #      )
    # print("------------------------------------")
    
    # use the fitted parameters in supergaus function to calculate z-values for plotting
    Z_true = super_gaussian_2d((x_units, y_units), A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, n_fit, offset_fit).reshape(np.max(data[1]) + 1, np.max(data[1]) + 1)

    # set axis boundaries to 6 sigma after fitting for nice plotting
    lower_x = x0_fit-6*sigma_x_fit  #lower x value in cm for plotting
    upper_x = x0_fit+6*sigma_x_fit  #upper x value in cm for plotting
    lower_y = y0_fit-6*sigma_y_fit  #lower x value in cm for plotting
    upper_y = y0_fit+6*sigma_y_fit  #upper x value in cm for plotting

    # clean data from zeros because we assume error=sqrt(data2) and divide by this when doing red. chi2
    def clean_and_chi2(data1,data2): #data2 = the one with error on its values
        data1_clean = []
        data2_clean = []
        for i in range(len(data1)):
            for j in range(len(data1[0])):
                if data1[i][j] > 0 and data2[i][j]>0:
                    data1_clean += [data1[i][j]]
                    data2_clean += [data2[i][j]]

        data1_clean = np.asarray(data1_clean).flatten()
        data2_clean = np.asarray(data2_clean).flatten()
        
        #calculate reduced chi2
        chi2 = np.sum( (data1_clean - data2_clean)**2 / np.sqrt(data2_clean)**2 ) 
        dofs = len(data1_clean) - len(popt)
        red_chi = chi2/dofs
        return(red_chi,chi2)
    
    red_chi_SG, chi2_SG = clean_and_chi2(Z_true,result)
    print("Supergaussian red. chi2 is: " + str(round(red_chi_SG,2)))

    red_chi_RawReConv, chi2_RawReConv = clean_and_chi2(matrix(data),reconvolved_norm)
    print("Raw-ReConvolved chi2 is: " + str(round(chi2_RawReConv,2)))


    fig = plt.figure(figsize=plt.figaspect(0.5))
    
    #RAW DATA as contour plot
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.title.set_text("Raw Data - Countour Plot")
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

    #DECONVOLVED DATA as contour plot
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.title.set_text("Deconvolved Data - Contour Plot")
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

    #RE-CONVOLCVED DATA as contour plot
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
    # ax.plot([],[],' ', label=f'$red. \\chi^2=${red_chi_SG:.3f}' )

    # Set legend for ax and labels
    ax.legend(loc='best',handlelength=0, handletextpad=0)
    ax.set_zlim(0,1.1)
    ax.set_xlim(lower_x,upper_x)
    ax.set_ylim(lower_y,upper_y)
    ax.set_xlabel("x / cm")
    ax.set_ylabel("y / cm")
    ax.set_zlabel("normalised intensity")

    plt.show()
