import os

os.chdir('../')
script_to_execute = 'python3 train.py --logs_file_path ../results/debugging/debugging.csv --model XGBoost_Classifier --dataset_path ../built_dataset/only_hospitalized.csv --balancing Oversampling --feature_retrieval SequentialSelector --splitting Holdout --test_size 0.2 --nofsplits 5 --noftopfeatures 28 --preprocessing No --store_features_selection_report'
os.system(script_to_execute)
