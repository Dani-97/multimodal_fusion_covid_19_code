import os

os.chdir('../')
script_to_execute = 'python3 train.py --logs_file_path ../results/debugging/debugging.csv --model SVM_Classifier --dataset_path ../built_dataset/hospitalized_and_urgencies.csv --balancing Oversampling --feature_retrieval No --splitting Holdout --test_size 0.2 --nofsplits 5 --noftopfeatures 5 --preprocessing Standardization --store_features_selection_report'
os.system(script_to_execute)
