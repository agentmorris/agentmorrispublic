#
# Render the shared spreadsheet about aerial/drone wildlife datasets to markdown
#

#%% Imports and constants

import pandas as pd
import humanfriendly

input_file = r'g:\temp\Aerial_drone wildlife metadata curation party - drone_aerial wildlife datasets.csv'
output_file = r'g:\temp\drone-datasets-body.md'


#%% Read input annotations, write output

df = pd.read_csv(input_file)

output_lines = []

starting_header_level = '###'

# i_row = 0; row = df.iloc[i_row]
for i_row,row in df.iterrows():

    # Ignore everything below this
    if not isinstance(row['Assigned to'],str):
        break
    
    s = ''
    s += starting_header_level + ' ' + row['Name'] + '\n\n'
    
    s += row['Short description'] + '\n'
    
    s += '  \n'
    
    if isinstance(row['Citation'],str):
        s += row['Citation'] + '\n\n'
        
    # s += '<a href="{}">{}</a>\n'.format(row['URL'],row['URL'])
    
    size_bytes = 1000*1000*1000*float(row['Size in GB'])
    size_string = humanfriendly.format_size(size_bytes)
    
    s += '* {}, downloadable via {} from {} (<a href="{}">download link</a>)\n'.format(
        size_string, row['Download mechanism'], row['Hosting site'], row['URL'])
    
    metadata_addendum = ''
    if isinstance(row['Metadata standard'],str):
        metadata_addendum = ' (' + row['Metadata standard'] + ')'
    s += '* Metadata in {} format{}\n'.format(row['Metadata raw format'],metadata_addendum)
    
    s += '* Categories: {}\n'.format(row['Categories/species'])
    s += '* Vehicle type: {}\n'.format(row['Vehicle type'])
    
    image_addendum = ''
    if isinstance(row['Image notes'],str):
        note = row['Image notes']
        note = note[0].lower() + note[1:]
        image_addendum = ' (' + note + ')'
        
    s += '* Image information: {} {} images{}\n'.format(row['Number of images or videos'],row['Channels'],image_addendum)
    
    s += '* Annotation information: {} {}\n'.format(int(row['Number of annotations']),row['Annotation type'])
    s += '* Typical animal size in pixels: {}\n'.format(row['Typical animal width in pixels'])
    
    if isinstance(row['Sample code'],str):
        
        code_url = row['Sample code']
        code_link_name = code_url.split('/')[-1]
        
        s += '* Code to render sample annotated image: <a href="{}">{}</a>\n'.format(
            row['Sample code'],code_link_name)
    
    s += '  \n'
    s += '  \n'
    
    s += '<img src="{}" width=700>\n'.format(row['Sample image'])
    
    s += '  \n'
    s += '  \n'
    
    output_lines.append(s)

with open(output_file,'w') as f:
    for s in output_lines:
        f.write(s)
        
        