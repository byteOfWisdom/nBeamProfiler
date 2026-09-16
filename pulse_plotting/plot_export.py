from matplotlib import pyplot as plt
import matplotlib
import os

def PGF_plots():
    # activate pgf-plotting
    matplotlib.use("pgf")
    # print('Using the ' + matplotlib.get_backend() + ' backend') #for debugging

    #update parameter for the right output font/size
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif", # use serif/main font for text elements
        # 'font.size': 10,        # change text size
        "pgf.rcfonts": False,   # don't setup fonts from rc parameters
        'figure.autolayout': True,
        # "text.usetex": True,     # use inline math for ticks - THIS BREAKS THE EXPORT!
    })

    # change figure size for all figures plotted
    # LaTeX textwidth is 4.7747 inches and textheight is 9.3611 inches
    # divide textheight by 3 so that two figures + captions fit on one page
    # plt.rcParams["figure.figsize"] = (4.7747, 9.3611/3.5)
    # plt.rcParams["figure.figsize"] = (4.7747/2, 9.3611/3.5)
    return 

def Save_Plot(path, title):
    #check if PGF is used and save plot
    if matplotlib.get_backend() == 'pgf':
        plt.savefig(path + title +'.pgf', format='pgf')
        print('Plot saved as PGF')
    else:
        plt.savefig(path + title +'.pdf')
        print('Plot saved as PDF')

def Scriptpath(file):
    # path to where 'file' is
    scriptpath = str(os.path.abspath(os.path.dirname(file))) + '/'
    # print("Script path is:") #for debugging
    # print(scriptpath) #for debugging
    return scriptpath