import nibabel as nib
import matplotlib.pylab as plt
from os import listdir
from os.path import isfile, join

def filesInDirectory(print_files=False):
    onlyfiles = [f for f in listdir('.\\images') if isfile(join('.\\images', f))]
    file_array = []
    incorrect_files = []
    for x in onlyfiles:
        if x[-4:] == '.nii':
            file_array.append(x)
        else:
            incorrect_files.append(x)
    
    if len(onlyfiles) != len(file_array):
        print('\nThe following files did not match the correct file type for processing [nii]:\n')
        print(incorrect_files)
        print_files = True
    
    if print_files: 
        print('\nThe program will proceed with analyzing the following files:\n')
        print(file_array)
        

    return file_array



def openNii(path):
    img = nib.load('.\\images\\'+ path).get_fdata()
    img.shape

    #print(f"The .nii files are stored in memory as numpy's: {type(img)}.")

    plt.style.use('default')
    fig, axes = plt.subplots(4,4, figsize=(12,12))

    for i, ax in enumerate(axes.reshape(-1)):
        ax.imshow(img[:,:,1 + i])
        
    plt.savefig('.\\export\\'+path+'.png')


if __name__ == '__main__':
    files = filesInDirectory(True)

    for x in files:
        openNii(x)
