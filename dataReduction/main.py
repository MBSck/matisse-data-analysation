from show_allred import show_allred



if __name__ == "__main__":
    datadir=r'/path/to/drs/products/'
    outputdir = datadir + 'plots/'
    show_allred(inputdir, outputdir=inputdir + '/plots/', nbProc=6)
