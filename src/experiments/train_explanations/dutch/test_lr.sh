
# for i in $(seq 1 2);
# do
#     uv run python ../../../fire360/explanations/compute_explanations.py --dataset_name dutch --model_name dutch_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/fire360/artifacts/dutch/synthetic_data/synthetic_data_100000_epochs_2500_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/fire360/artifacts/dutch/bb/dutch_BB.pth  --debug True --num_processes 20 --explanation_type logistic --validation_seed $i --store_path /home/lcorbucci/fire360/artifacts/dutch/explanations
# done


for i in $(seq 50 51);
do
    uv run python ../../../fire360/explanations/compute_explanations.py --dataset_name dutch --model_name dutch_BB --top_k 2500 --synthetic_dataset_path /home/lcorbucci/fire360/artifacts/dutch/synthetic_data/synthetic_data_100000_epochs_2500_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/fire360/artifacts/dutch/bb/dutch_BB.pth  --debug True --num_processes 20 --explanation_type logistic --validation_seed $i --store_path /home/lcorbucci/fire360/artifacts/dutch/explanations
done

for i in $(seq 52 53);
do
    uv run python ../../../fire360/explanations/compute_explanations.py --dataset_name dutch --model_name dutch_BB --top_k 5000 --synthetic_dataset_path /home/lcorbucci/fire360/artifacts/dutch/synthetic_data/synthetic_data_100000_epochs_2500_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/fire360/artifacts/dutch/bb/dutch_BB.pth  --debug True --num_processes 20 --explanation_type logistic --validation_seed $i --store_path /home/lcorbucci/fire360/artifacts/dutch/explanations
done