import os

os.chdir('../')
script_to_execute = 'python3 train.py --logs_file_path ../results/debugging/debugging.csv --model XGBoost_Classifier --dataset_path ../built_dataset/debugging_radiomics.csv --balancing Oversampling --feature_retrieval Fisher --splitting Holdout --test_size 0.2 --nofsplits 5 --noftopfeatures 121 --preprocessing No --store_features_selection_report --manual_seed 8'
os.system(script_to_execute)
