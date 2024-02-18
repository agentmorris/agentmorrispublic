#%% Constants and imports

"""
# Installation on Windows
#
# Python 3.10 is the latest version that's compatible with TF 2.10, which is the 
# last version with native GPU windows support.
#
# https://www.tensorflow.org/install/pip#windows-native

mamba create -n opensoundscape python=3.10 pip
mamba activate opensoundscape

# This was enough for TF to report the GPU as being prsent, but when I actually tried
# to run inference, I got:
#
# libdevice not found at ./libdevice.10.bc
#
# ...until I *also* installed the system CUDA 11.2 from:
#
# https://developer.nvidia.com/cuda-11.2.0-download-archive
mamba install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install "tensorflow<2.11"
pip install opensoundscape

# OpenSoundscape forces installation of a newer version of protobuf, which makes older 
# versions of TF sad, but for this test, we're not using features of OpenSoundscape that 
# require it, so force installation of 3.19.6 again.
pip install protobuf==3.19.6
pip install tensorflow_hub
mamba install -c conda-forge spyder

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
"""

import os
import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
import tensorflow as tf

# Just testing installation, this is not imported directly
import opensoundscape # noqa

# Just testing installation, this is not imported directly
import tensorflow_hub # noqa

# data_folder = r'G:\temp\perch-stuff\audiomoth\audiomoth'
data_folder = r'G:\temp\audio'
assert os.path.isdir(data_folder)


#%% Create model

model = torch.hub.load('kitzeslab/bioacoustics-model-zoo', 'Perch', trust_repo=True)


#%% Enumerate .wav files

audio_files = os.listdir(data_folder)
audio_files = [fn for fn in audio_files if (fn.lower().endswith('.wav')) or (fn.lower().endswith('.mp3'))]
audio_files = [os.path.join(data_folder,fn) for fn in audio_files]

print('Enumerated {} data files:'.format(len(audio_files)))
for fn in audio_files:
    print(fn)


#%% Run inference, write results for each .wav file to .csv

print("GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# i_file = 0; fn = audio_files[i_file]
for i_file,fn in enumerate(audio_files):
    
    print('\n*** Running inference for file {} of {}: {} ***\n'.format(
        i_file,len(audio_files),fn))
    predictions = model.predict([fn]) 
    output_file = os.path.splitext(fn)[0] + '.csv'
    print('Writing results to {}'.format(output_file))
    predictions.to_csv(output_file,header=True,index=True,float_format='%.5g')
    # embeddings = model.generate_embeddings([fn])
    

#%% Find detections above a threshold

csv_files = os.listdir(data_folder)
csv_files = [fn for fn in csv_files if fn.lower().endswith('.csv') and 'BirdNET' not in fn]
csv_files = [os.path.join(data_folder,fn) for fn in csv_files]

print('Enumerated {} csv files'.format(len(csv_files)))

metadata_columns = ['file', 'start_time', 'end_time']

# i_file = 0; fn = csv_files[i_file]
for i_file,fn in enumerate(csv_files):
    
    print('Analyzing results in {}'.format(fn))

    df = pd.read_csv(fn)
    
    # The scores don't appear to be log-probabilities, but for this simple test,
    # it doesn't really matter what they are, I'll just normalize by min/max.
    df_numeric = df.drop(columns=metadata_columns)
    max_val = np.max(df_numeric.max())
    min_val = np.min(df_numeric.min())
    
    df_numeric = (df_numeric - min_val) / (max_val - min_val)
            
    threshold = 0.5
    
    # This is wildly inefficienct, but I'm not after efficiency here, I just want
    # some detections.
    # row = df.iloc[0]
    for i_row,row in tqdm(df_numeric.iterrows(),total=len(df_numeric)):
        for c in df_numeric.columns:
            if row[c] >= threshold:
                t = df.iloc[i_row]['start_time']
                print('{} at {} ({})'.format(c,t,row[c]))
