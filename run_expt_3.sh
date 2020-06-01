random_states=(1 2 3 4 5 6 7 8 9 10)

export NEPTUNE_API_TOKEN=$2

all_datasets=jha_bug_bin_2017,scalabrino_bug_bin_2017,maalej_bug_bin_2016,williams_bug_bin_2017,di_sorbo_bug_bin_2017,guzman_bug_bin_2015,tizard_bug_bin_2019

# Run experiment 3 on all datasets for 10 different random seeds
for random_state in ${random_states[@]}; do

				echo "Now running dataset $all_datasets with zero_shot_datasets $all_datasets in random state $random_state"

        python software_feedback_cls.py --dataset_list=$all_datasets --random_state=$random_state --zero_shot_label=bug --zero_shot_dataset_list=$all_datasets --output_text --do_classical --num_epochs=60 --num_fine_tuning_epochs=60 --early_stopping_patience=10 --batch_size_train=32 --batch_size_eval=32 --max_length=128 --neptune_username=$1
done

# Collect the generated output files and output them to a meta-experiment on Neptune
python overall_experiment_data_collector.py --experiment_number=3 --experiment_name=experiment_3_all_datasets --neptune_username=$1

#python software_feedback_cls.py --dataset_list=jha_2017,scalabrino_2017,maalej_2016,williams_2017,guzman_2015,tizard_2019 --output_text --zero_shot_dataset_list=jha_2017,scalabrino_2017,maalej_2016,williams_2017,di_sorbo_2017,guzman_2015,tizard_2019 --num_epochs=60 --num_fine_tuning_epochs=60 --batch_size_train=8 --batch_size_eval=8 --max_length=128 --zero_shot_label=bug --neptune_username=$1
