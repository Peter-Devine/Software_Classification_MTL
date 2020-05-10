datasets=(jha_bug_bin_2017 scalabrino_bug_bin_2017 maalej_bug_bin_2016 williams_bug_bin_2017 di_sorbo_bug_bin_2017 guzman_bug_bin_2015 tizard_bug_bin_2019)
random_states=(1 2 3 4 5 6 7 8 9 10)

export NEPTUNE_API_TOKEN=$2

# Delete all output files from previous runs
python output_clean.py

# Run experiment 1 on all datasets for 10 different random seeds
for dataset in ${datasets[@]}; do
	for random_state in ${random_states[@]}; do
					echo "Now running dataset $dataset in random state $random_state"

	        python software_feedback_cls.py --dataset_list=$dataset --random_state=$random_state --output_text --do_classical --num_epochs=60 --num_fine_tuning_epochs=60 --early_stopping_patience=10 --batch_size_train=32 --batch_size_eval=32 --max_length=128 --neptune_username=$1
	done
done

# Collect the generated output files and output them to a meta-experiment on Neptune
python overall_experiment_data_collector.py --experiment_number=1 --experiment_name=experiment_1_all_datasets --neptune_username=$1
