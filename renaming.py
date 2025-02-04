import os
import re

# Define the directory containing the files
directory = r'M:\Documents\PhD\Code\PhD_Projects\IFE_degradation\Results\20241015142936\18_9_geo_smoothed\Opt_model_results'  # Change this to the correct directory path if needed

# Define the pattern to match the filenames, allowing for decimals in Y
pattern = re.compile(r'(\d+.?\d*)MWh_(\d+.?\d*)MW_(\d+T)_1.0_Results')

# Iterate over the files in the directory
count = 0
files_changed_count = 0
for filename in os.listdir(directory):
    # Full path to the current file
    file_path = os.path.join(directory, filename)
    print(f'count = {count}')
    # Ensure it's a file
    if os.path.isfile(file_path):
        # Match the pattern
        match = pattern.match(filename)
        if match:
            # Extract the relevant parts from the match groups
            X = match.group(1)  # XMWh
            Y_charge = match.group(2)  # Y before 'c', possibly a decimal
            # Y_discharge = match.group(3) # Y before 'd', possibly a decimal
            T = match.group(3)  # Variable part before 'T'
            # assert Y_charge == Y_discharge, "Different charging and discharging rates"
            # Construct the new filename
            new_filename = f'{X}MWh_{Y_charge}MW_{T}_1.1_Results'
            # Full path to the new file
            new_file_path = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(file_path, new_file_path)
            print(f'Renamed: {filename} to {new_filename}')
            files_changed_count += 1
            print(f'Files changed count: {files_changed_count}')
        else:
            print(f'No match found: {filename}')
            print(f'Files changed count: {files_changed_count}')
        count += 1