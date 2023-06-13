#
# gdal_convert_nan_to_nodata.py
#
# For a folder of .tif files, converting NaN values to a specified NODATA value, and writing
# that as the NODATA value in metadata for each file.
#

#%% Notes to self

#
# To view statistics for an image:
#
# gdalinfo input.tif -stats --config GDAL_PAM_ENABLED NO
#
# To build a VRT on the outputs:
#
# gdalbuildvrt x.vrt *x*.tif
# gdalbuildvrt y.vrt *y*.tif
# gdalbuildvrt -overwrite -resolution average -separate -r nearest mosaic.vrt *prob*.vrt 
#
# To run this script on a folder:
#
# python gdal_convert_nan_to_nodata.py g:\temp\mosaics g:\temp\nodata_mosaics
#
# The gist of what this script does on each folder:
#    
# Formerly:
#
# python "gdal_calc.py" --co="COMPRESS=LZW" --co="PREDICTOR=2" -A "input.tif" --outfile="output.tif" --calc="nan_to_num(A, nan=-99)" --NoDataValue=-99
#
# Now:
#
# gdalwarp -srcnodata nan -dstnodata -99 -co COMPRESS=LZW -co PREDICTOR=2 "input.tif" "output.tif"
#
    
#%% Imports and constants

import subprocess
import os

default_nodata_val = -9999

default_n_threads = 4

gdal_calc_location = None


#%% Main function(s)

def configure_gdal_calc():
    """
    No longer used, since I switched from gdal_calc.py to gdalwarp, but this may be handy in the
    future, so keeping it around.
    """
    
    global gdal_calc_location
    
    if gdal_calc_location is not None:
        return
    
    # gdal_calc_folder = r'C:\anaconda3\envs\gdal\Scripts'

    # E.g. 'C:\\anaconda3\\envs\\gdal\\Lib\\site-packages\\osgeo
    from osgeo import gdal
    gdal_location = gdal.__file__
    assert 'Lib' in gdal_location
    gdal_calc_folder = os.path.join(gdal_location.split('Lib')[0],'Scripts')

    gdal_calc_location = os.path.join(gdal_calc_folder,'gdal_calc.py')
    assert os.path.isfile(gdal_calc_location)
    

def _process_file(input_fn,output_fn=None,nodata_val=default_nodata_val):
    """
    Convert NaN to a valid NODATA value for one file, with the full input path specified.
    """
    
    assert input_fn.lower().endswith('.tif'), 'I only know how to process GeoTIFF files'
    
    if output_fn is None:
        tokens = os.path.splitext(input_fn)
        output_fn = tokens[0] + '_nodata_replaced' + tokens[1]
        
    print('Processing {} to {}'.format(input_fn,output_fn))
    
    cmd = 'gdalwarp -srcnodata nan -dstnodata {} -co COMPRESS=LZW -co PREDICTOR=2 "{}" "{}"'.format(
        nodata_val,input_fn,output_fn)

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if len(result.stderr) > 0:
        print('Warning: GDAL error info: {}'.format(result.stderr))
    
    print('GDAL output:')
    print(result.stdout)


def process_file(basename,input_folder,output_folder,nodata_val=default_nodata_val):
    """
    Convert NaN to a valid NODATA value for one file, with the relative input path and output
    folder name specified.
    """
    
    input_fn = os.path.join(input_folder,basename)
    assert os.path.isfile(input_fn)
    
    tokens = os.path.splitext(basename)
    output_fn = os.path.join(output_folder,tokens[0] + '_nodata_replaced' + tokens[1])
    
    _process_file(input_fn,output_fn,nodata_val)
    
    
def process_folder(input_folder,output_folder,nodata_val=default_nodata_val,n_threads=default_n_threads):
    """
    Convert NaN to a valid NODATA value for all .tif files in a folder.
    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        assert os.path.isdir(output_folder), '{} is not a folder'.format(output_folder)
        
    files = os.listdir(input_folder)
    input_tif_files = [fn for fn in files if fn.lower().endswith('.tif')]
    
    if (n_threads <= 1):
        
        for basename in input_tif_files:        
            process_file(basename,input_folder,output_folder,nodata_val)    
        
    else:
        
        from functools import partial
        from multiprocessing.pool import ThreadPool as ThreadPool
        from multiprocessing.pool import Pool as Pool
       
        use_threads = True
       
        n_workers = min(n_threads,len(input_tif_files))
       
        if use_threads:
          print('Starting parallel thread pool with {} workers'.format(n_workers))
          pool = ThreadPool(n_threads)
        else:
          print('Starting parallel process pool with {} workers'.format(n_workers))
          pool = Pool(n_threads)
     
        _ = list(pool.map(partial(process_file,
                                        input_folder=input_folder,
                                        output_folder=output_folder,
                                        nodata_val=nodata_val),input_tif_files))

      
#%% Command-line driver

import argparse
import sys

def main():
    
    parser = argparse.ArgumentParser(
        description='Convert NaN to NODATA in a folder of GeoTIFF files')
    parser.add_argument(
        'input_folder',
        help='Path to a folder containing GeoTIFF files')
    parser.add_argument(
        'output_folder',
        help='Path to write output files')
    parser.add_argument(
        '--nodata_val',
        type=int,
        default=default_nodata_val,
        help='NODATA val to write to output files (defaults to {})'.format(default_nodata_val))
    parser.add_argument(
        '--n_threads',
        type=int,
        default=default_n_threads,
        help='Number of workers (defaults to {})'.format(default_n_threads))
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    process_folder(args.input_folder,args.output_folder,args.nodata_val,args.n_threads)

if __name__ == '__main__':
    main()
