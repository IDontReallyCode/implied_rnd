import pickle
import bz2
import os
import glob

mainfolder = 'C:/Users/p_h_d/Dropbox/00_INVESTING/Data/'

def main():
    # Get a list of all folders in the root path
    # folders = [f for f in glob.glob(mainfolder + "**/", recursive=True)]
    folders = next(os.walk(mainfolder))[1]

    # Sort the list of folders
    folders.sort()
    
    for thisdate in folders:
        print(thisdate)
        thisfile = mainfolder + thisdate + '/AMZN' + thisdate + '.pbz2'
        try:
            f = bz2.BZ2File(thisfile, 'r')
            OC = pickle.load(f)
        except:
            pass
        pass
    pass




if __name__ == '__main__':
    main()
