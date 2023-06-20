# Hospitalized And Urgencies.
# 
# python3 build_dataset_scenario_III.py --images_dir_root ../original_dataset/distributed_images/ --input_csv_file_path ../original_dataset/original_input_preprocessed.csv --approach VGG_16_Model_Only_Hospitalized --layer layer_fc6 --device CUDA --output_path ../built_dataset/only_hospitalized_vgg_16_fc6.csv --dataset_version simplified --imaging_features_csv_path ../built_dataset/vgg_16_fc6.csv
# 
# python3 build_dataset_scenario_III.py --images_dir_root ../original_dataset/distributed_images/ --input_csv_file_path ../original_dataset/original_input_preprocessed.csv --approach VGG_16_Model_Only_Hospitalized --layer layer_fc7 --device CUDA --output_path ../built_dataset/only_hospitalized_vgg_16_fc7.csv --dataset_version simplified --imaging_features_csv_path ../built_dataset/vgg_16_fc7.csv
# 
# python3 build_dataset_scenario_III.py --images_dir_root ../original_dataset/distributed_images/ --input_csv_file_path ../original_dataset/original_input_preprocessed.csv --approach VGG_16_Model_Only_Hospitalized --layer layer_fc8 --device CUDA --output_path ../built_dataset/only_hospitalized_vgg_16_fc8.csv --dataset_version simplified --imaging_features_csv_path ../built_dataset/vgg_16_fc8.csv
# 
# Only Hospitalized.
# 
# python3 build_dataset_scenario_III.py --images_dir_root ../original_dataset/distributed_images/ --input_csv_file_path ../original_dataset/original_input_preprocessed.csv --approach VGG_16_Model_Hospitalized_And_Urgencies --layer layer_fc6 --device CUDA --output_path ../built_dataset/hospitalized_and_urgencies_vgg_16_fc6.csv --dataset_version simplified --imaging_features_csv_path ../built_dataset/vgg_16_fc6.csv
# 
# python3 build_dataset_scenario_III.py --images_dir_root ../original_dataset/distributed_images/ --input_csv_file_path ../original_dataset/original_input_preprocessed.csv --approach VGG_16_Model_Hospitalized_And_Urgencies --layer layer_fc7 --device CUDA --output_path ../built_dataset/hospitalized_and_urgencies_vgg_16_fc7.csv --dataset_version simplified --imaging_features_csv_path ../built_dataset/vgg_16_fc7.csv
# 
# python3 build_dataset_scenario_III.py --images_dir_root ../original_dataset/distributed_images/ --input_csv_file_path ../original_dataset/original_input_preprocessed.csv --approach VGG_16_Model_Hospitalized_And_Urgencies --layer layer_fc8 --device CUDA --output_path ../built_dataset/hospitalized_and_urgencies_vgg_16_fc8.csv --dataset_version simplified --imaging_features_csv_path ../built_dataset/vgg_16_fc8.csv

# Only imaging features - Hospitalized And Urgencies.

python3 build_dataset_scenario_II.py --images_dir_root ../original_dataset/distributed_images/ --input_csv_file_path ../original_dataset/original_input_preprocessed.csv --approach VGG_16_Model_Hospitalized_And_Urgencies --layer layer_fc6 --device CUDA --output_path ../built_dataset/hospitalized_and_urgencies_vgg_16_fc6_only_imaging_features.csv --dataset_version simplified --precomputed_imaging_features_csv_path ../built_dataset/vgg_16_fc6.csv

python3 build_dataset_scenario_II.py --images_dir_root ../original_dataset/distributed_images/ --input_csv_file_path ../original_dataset/original_input_preprocessed.csv --approach VGG_16_Model_Hospitalized_And_Urgencies --layer layer_fc7 --device CUDA --output_path ../built_dataset/hospitalized_and_urgencies_vgg_16_fc7_only_imaging_features.csv --dataset_version simplified --precomputed_imaging_features_csv_path ../built_dataset/vgg_16_fc7.csv

python3 build_dataset_scenario_II.py --images_dir_root ../original_dataset/distributed_images/ --input_csv_file_path ../original_dataset/original_input_preprocessed.csv --approach VGG_16_Model_Hospitalized_And_Urgencies --layer layer_fc8 --device CUDA --output_path ../built_dataset/hospitalized_and_urgencies_vgg_16_fc8_only_imaging_features.csv --dataset_version simplified --precomputed_imaging_features_csv_path ../built_dataset/vgg_16_fc8.csv

# Only imaging features - Only Hospitalized.

python3 build_dataset_scenario_II.py --images_dir_root ../original_dataset/distributed_images/ --input_csv_file_path ../original_dataset/original_input_preprocessed.csv --approach VGG_16_Model_Only_Hospitalized --layer layer_fc6 --device CUDA --output_path ../built_dataset/only_hospitalized_vgg_16_fc6_only_imaging_features.csv --dataset_version simplified --precomputed_imaging_features_csv_path ../built_dataset/vgg_16_fc6.csv

python3 build_dataset_scenario_II.py --images_dir_root ../original_dataset/distributed_images/ --input_csv_file_path ../original_dataset/original_input_preprocessed.csv --approach VGG_16_Model_Only_Hospitalized --layer layer_fc7 --device CUDA --output_path ../built_dataset/only_hospitalized_vgg_16_fc7_only_imaging_features.csv --dataset_version simplified --precomputed_imaging_features_csv_path ../built_dataset/vgg_16_fc7.csv

python3 build_dataset_scenario_II.py --images_dir_root ../original_dataset/distributed_images/ --input_csv_file_path ../original_dataset/original_input_preprocessed.csv --approach VGG_16_Model_Only_Hospitalized --layer layer_fc8 --device CUDA --output_path ../built_dataset/only_hospitalized_vgg_16_fc8_only_imaging_features.csv --dataset_version simplified --precomputed_imaging_features_csv_path ../built_dataset/vgg_16_fc8.csv
