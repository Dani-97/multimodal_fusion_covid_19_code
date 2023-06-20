import os

os.chdir('../')
script_to_execute = 'python3 train.py --logs_file_path ../results_debugging/debugging.csv --experiment_name debugging --model XGBoost_Classifier --dataset_path ../built_dataset/hospitalized_and_urgencies_vgg_16_fc6.csv --preprocessing Normalization --manual_seeds 0 1 2 3 4 --imputation No_Imputation_Model --balancing Oversampling --csv_path_with_attrs_types ../original_dataset/attrs_headers_types.csv --feature_retrieval MutualInformation --store_features_selection_report --splitting Holdout --noftopfeatures all --nofsplits 5 --test_size 0.2 --n_neighbors 5 '
os.system(script_to_execute)

