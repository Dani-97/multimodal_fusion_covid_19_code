import os

os.chdir('../')
script_to_execute = 'python3 train.py --logs_file_path ../results_debugging/debugging_fc6_20.csv --experiment_name debugging_fc6_20 --model XGBoost_Classifier --dataset_path ../results_debugging/debugging_dataset_vgg16_fc6.csv --preprocessing Normalization --manual_seeds 0 1 2 3 4 --imputation No_Imputation_Model --balancing Oversampling --csv_path_with_attrs_types ../original_dataset/attrs_headers_types.csv --feature_retrieval MutualInformation --store_features_selection_report --splitting Holdout --noftopfeatures 20 --nofsplits 5 --test_size 0.2 --n_neighbors 5'
os.system(script_to_execute)

