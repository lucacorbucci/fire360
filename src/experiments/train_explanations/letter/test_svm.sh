
for i in $(seq 2 2);
do
    uv run python ../../../fire360/explanations/compute_explanations.py --dataset_name letter --model_name letter_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/fire360/artifacts/letter/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/fire360/artifacts/letter/bb/letter_BB.pth  --debug True --num_processes 20 --explanation_type svm --validation_seed $i --store_path /home/lcorbucci/fire360/artifacts/letter/explanations
done


# for i in $(seq 50 51);
# do
#     uv run python ../../../fire360/explanations/compute_explanations.py --dataset_name letter --model_name letter_BB --top_k 2500 --synthetic_dataset_path /home/lcorbucci/fire360/artifacts/letter/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/fire360/artifacts/letter/bb/letter_BB.pth  --debug True --num_processes 20 --explanation_type svm --validation_seed $i --store_path /home/lcorbucci/fire360/artifacts/letter/explanations
# done

# for i in $(seq 52 53);
# do
#     uv run python ../../../fire360/explanations/compute_explanations.py --dataset_name letter --model_name letter_BB --top_k 5000 --synthetic_dataset_path /home/lcorbucci/fire360/artifacts/letter/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/fire360/artifacts/letter/bb/letter_BB.pth  --debug True --num_processes 20 --explanation_type svm --validation_seed $i --store_path /home/lcorbucci/fire360/artifacts/letter/explanations
# done
