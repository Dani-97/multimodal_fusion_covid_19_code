import os

os.chdir('../')
script_to_execute = 'python3 train.py --logs_file_path ../results/debugging.csv --experiment_name debugging --model XGBoost_Classifier --dataset_path ~/Downloads/debugging.csv --preprocessing Normalization --manual_seeds 0 1 2 3 4 --imputation No_Imputation_Model --balancing Oversampling --csv_path_with_attrs_types ../original_dataset/attrs_headers_types.csv --feature_retrieval MutualInformation --store_features_selection_report --splitting Holdout --noftopfeatures 50 --nofsplits 5 --test_size 0.2 --n_neighbors 5 '
os.system(script_to_execute)

script_to_execute = 'python3 test.py --logs_file_path ../results/debugging_0_75.csv --experiment_name debugging --model XGBoost_Classifier --dataset_path ~/Downloads/debugging.csv --balancing Oversampling --feature_retrieval MutualInformation --splitting Holdout --test_size 0.2 --nofsplits 5 --noftopfeatures 50 --preprocessing Normalization --manual_seeds 0 1 2 3 4 --store_features_selection_report --csv_path_with_attrs_types ../original_dataset/attrs_headers_types.csv --imputation No_Imputation_Model --model_path ../results/debugging_XGBoost_Classifier_50_MutualInformation_Oversampling_model --operation_point 0.75'
os.system(script_to_execute)

script_to_execute = 'python3 test.py --logs_file_path ../results/debugging_0_6.csv --experiment_name debugging --model XGBoost_Classifier --dataset_path ~/Downloads/debugging.csv --balancing Oversampling --feature_retrieval MutualInformation --splitting Holdout --test_size 0.2 --nofsplits 5 --noftopfeatures 50 --preprocessing Normalization --manual_seeds 0 1 2 3 4 --store_features_selection_report --csv_path_with_attrs_types ../original_dataset/attrs_headers_types.csv --imputation No_Imputation_Model --model_path ../results/debugging_XGBoost_Classifier_50_MutualInformation_Oversampling_model --operation_point 0.6'
os.system(script_to_execute)

script_to_execute = 'python3 test.py --logs_file_path ../results/debugging_0_8.csv --experiment_name debugging --model XGBoost_Classifier --dataset_path ~/Downloads/debugging.csv --balancing Oversampling --feature_retrieval MutualInformation --splitting Holdout --test_size 0.2 --nofsplits 5 --noftopfeatures 50 --preprocessing Normalization --manual_seeds 0 1 2 3 4 --store_features_selection_report --csv_path_with_attrs_types ../original_dataset/attrs_headers_types.csv --imputation No_Imputation_Model --model_path ../results/debugging_XGBoost_Classifier_50_MutualInformation_Oversampling_model --operation_point 0.8'
os.system(script_to_execute)

script_to_execute = 'python3 test.py --logs_file_path ../results/debugging_0_9.csv --experiment_name debugging --model XGBoost_Classifier --dataset_path ~/Downloads/debugging.csv --balancing Oversampling --feature_retrieval MutualInformation --splitting Holdout --test_size 0.2 --nofsplits 5 --noftopfeatures 50 --preprocessing Normalization --manual_seeds 0 1 2 3 4 --store_features_selection_report --csv_path_with_attrs_types ../original_dataset/attrs_headers_types.csv --imputation No_Imputation_Model --model_path ../results/debugging_XGBoost_Classifier_50_MutualInformation_Oversampling_model --operation_point 0.9'
os.system(script_to_execute)
