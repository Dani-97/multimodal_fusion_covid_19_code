import os

os.chdir('../')
script_to_execute = 'python3 build_dataset.py --input_filename ../original_dataset/input.csv --headers_file ../original_dataset/headers.txt --output_path ../built_dataset/debugging.csv --approach Only_Hospitalized'
os.system(script_to_execute)
