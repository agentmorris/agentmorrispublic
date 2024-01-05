########
#
# character-encoding-test.py
#
# Print filenames that contain quirky character to the console, within and outside
# of a subprocess call.
#
########

#%% Imports and constants

import os,sys, locale

from md_utils.process_utils import execute_and_print
from md_utils.path_utils import find_images
from md_visualization import visualization_utils as vis_utils

base_folder = r"G:\temp\BarabObab"

# Argument to use for "errors" in Popen() when creating the child process
child_errors = None

# Argument to use for "encoding" in Popen() when creating the child process
child_encoding = None

# Custom environment variables to pass to the child proess
child_custom_environment = {'PYTHONIOENCODING':'utf-8'}


#%% Print files without a separate subprocess launch

def print_images(folder_name):
    
    all_images = find_images(folder_name,recursive=True,return_relative_paths=False)
    
    n_images_to_print = 10
    
    for i_image in range(0,n_images_to_print):
        
        fn = all_images[i_image]
        
        # Make sure we can print to the console
        print(fn)
        
        # Open the image to confirm that the filename is still valid, not just junk
        # we can print to the console
        im = vis_utils.open_image(fn)
        assert (im is not None) and (im.size[0] > 0) and (im.size[1] > 0)

    # ...for each image
    
# ...print_image(...)


#%% Run with and without a subprocess launch

def run_test(do_subprocess_launch=False):

    print('Console information:')
    print(sys.getdefaultencoding() + ' ' + str(locale.getdefaultlocale()))
    
    print_images(base_folder)

    if do_subprocess_launch:
        environ = os.environ.copy()
        if child_custom_environment is not None:
            for k in child_custom_environment:
                environ[k] = child_custom_environment[k]        
        execute_and_print('python character-encoding-test.py --do_subprocess_launch 0',
                          encoding=child_encoding,errors=child_errors,print_output=True,env=environ)
    

#%% Command-line driver

import argparse

def main():

    parser = argparse.ArgumentParser()        
    
    parser.add_argument('--do_subprocess_launch',type=int,default=1)
        
    args = parser.parse_args()
    run_test(do_subprocess_launch=args.do_subprocess_launch)
    
if __name__ == '__main__':
    main()
    
    
    