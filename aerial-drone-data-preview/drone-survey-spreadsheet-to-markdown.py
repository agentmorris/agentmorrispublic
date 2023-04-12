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

starting_header_level = '#'

# i_row = 0; row = df.iloc[i_row]
for i_row,row in df.iterrows():

    # Ignore everything below this
    if not isinstance(row['Assigned to'],str):
        break
    
    s = ''
    s += starting_header_level + ' ' + row['Name'] + '\n'
    s += row['Short description'] + '\n'
        
    s += starting_header_level + '#' + ' Download and format information' + '\n'
    
    size_bytes = 1000*1000*1000*float(row['Size in GB'])
    size_string = humanfriendly.format_size(size_bytes)
    
    s += '{}, downloadable via {} from {}\n'.format(size_string, row['Hosting site'],row['Download mechanism'])
    
    metadata_addendum = ''
    if isinstance(row['Metadata standard'],str):
        metadata_addendum = ' (' + row['Metadata standard'] + ')'
    s += 'Metadata in {} format{}\n'.format(row['Metadata format'],metadata_addendum)
    
    s += '<img src="{}" width=700>\n'.format(row['Sample image'])
    
    output_lines.append(s)
    
        
with open(output_file,'w') as f:
    for s in output_lines:
        f.write(s)
        
        