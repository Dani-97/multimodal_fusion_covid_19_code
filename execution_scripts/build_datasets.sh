images_dir_root=../../PROGNOSIS/original_dataset/distributed_images/
input_csv_file_path=../../PROGNOSIS/original_dataset/original_input_preprocessed.csv

cd ..

# Hospitalized And Urgencies - Only imaging data
python3 build_dataset_scenario_II.py --images_dir_root $images_dir_root --input_csv_file_path $input_csv_file_path --approach VGG_16_Model_Hospitalized_And_Urgencies --precomputed_imaging_features_csv_path ../built_dataset/vgg_16_fc6.csv --output_path ../built_dataset/hospitalized_and_urgencies_vgg_16_fc6_only_imaging_features.csv --layer layer_fc6 --dataset_version simplified --device CUDA
python3 build_dataset_scenario_II.py --images_dir_root $images_dir_root --input_csv_file_path $input_csv_file_path --approach VGG_16_Model_Hospitalized_And_Urgencies --precomputed_imaging_features_csv_path ../built_dataset/vgg_16_fc7.csv --output_path ../built_dataset/hospitalized_and_urgencies_vgg_16_fc7_only_imaging_features.csv --layer layer_fc7 --dataset_version simplified --device CUDA
python3 build_dataset_scenario_II.py --images_dir_root $images_dir_root --input_csv_file_path $input_csv_file_path --approach VGG_16_Model_Hospitalized_And_Urgencies --precomputed_imaging_features_csv_path ../built_dataset/vgg_16_fc8.csv --output_path ../built_dataset/hospitalized_and_urgencies_vgg_16_fc8_only_imaging_features.csv --layer layer_fc8 --dataset_version simplified --device CUDA

# Only Hospitalized - Only imaging data
python3 build_dataset_scenario_II.py --images_dir_root $images_dir_root --input_csv_file_path $input_csv_file_path --approach VGG_16_Model_Only_Hospitalized --precomputed_imaging_features_csv_path ../built_dataset/vgg_16_fc6.csv --output_path ../built_dataset/only_hospitalized_vgg_16_fc6_only_imaging_features.csv --layer layer_fc6 --dataset_version simplified --device CUDA
python3 build_dataset_scenario_II.py --images_dir_root $images_dir_root --input_csv_file_path $input_csv_file_path --approach VGG_16_Model_Only_Hospitalized --precomputed_imaging_features_csv_path ../built_dataset/vgg_16_fc7.csv --output_path ../built_dataset/only_hospitalized_vgg_16_fc7_only_imaging_features.csv --layer layer_fc7 --dataset_version simplified --device CUDA
python3 build_dataset_scenario_II.py --images_dir_root $images_dir_root --input_csv_file_path $input_csv_file_path --approach VGG_16_Model_Only_Hospitalized --precomputed_imaging_features_csv_path ../built_dataset/vgg_16_fc8.csv --output_path ../built_dataset/only_hospitalized_vgg_16_fc8_only_imaging_features.csv --layer layer_fc8 --dataset_version simplified --device CUDA

# Hospitalized And Urgencies - Clinical data + Imaging data
python3 build_dataset_scenario_III.py --images_dir_root $images_dir_root --input_csv_file_path $input_csv_file_path --approach VGG_16_Model_Hospitalized_And_Urgencies --imaging_features_csv_path ../built_dataset/vgg_16_fc6.csv --output_path ../built_dataset/hospitalized_and_urgencies_vgg_16_fc6.csv --layer layer_fc6 --dataset_version simplified --device CUDA
python3 build_dataset_scenario_III.py --images_dir_root $images_dir_root --input_csv_file_path $input_csv_file_path --approach VGG_16_Model_Hospitalized_And_Urgencies --imaging_features_csv_path ../built_dataset/vgg_16_fc7.csv --output_path ../built_dataset/hospitalized_and_urgencies_vgg_16_fc7.csv --layer layer_fc7 --dataset_version simplified --device CUDA
python3 build_dataset_scenario_III.py --images_dir_root $images_dir_root --input_csv_file_path $input_csv_file_path --approach VGG_16_Model_Hospitalized_And_Urgencies --imaging_features_csv_path ../built_dataset/vgg_16_fc8.csv --output_path ../built_dataset/hospitalized_and_urgencies_vgg_16_fc8.csv --layer layer_fc8 --dataset_version simplified --device CUDA

# Only Hospitalized - Clinical data + Imaging data
python3 build_dataset_scenario_III.py --images_dir_root $images_dir_root --input_csv_file_path $input_csv_file_path --approach VGG_16_Model_Only_Hospitalized --imaging_features_csv_path ../built_dataset/vgg_16_fc6.csv --output_path ../built_dataset/only_hospitalized_vgg_16_fc6.csv --layer layer_fc6 --dataset_version simplified --device CUDA
python3 build_dataset_scenario_III.py --images_dir_root $images_dir_root --input_csv_file_path $input_csv_file_path --approach VGG_16_Model_Only_Hospitalized --imaging_features_csv_path ../built_dataset/vgg_16_fc7.csv --output_path ../built_dataset/only_hospitalized_vgg_16_fc7.csv --layer layer_fc7 --dataset_version simplified --device CUDA
python3 build_dataset_scenario_III.py --images_dir_root $images_dir_root --input_csv_file_path $input_csv_file_path --approach VGG_16_Model_Only_Hospitalized --imaging_features_csv_path ../built_dataset/vgg_16_fc8.csv --output_path ../built_dataset/only_hospitalized_vgg_16_fc8.csv --layer layer_fc8 --dataset_version simplified --device CUDA
