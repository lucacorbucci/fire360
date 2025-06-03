
for i in $(seq 1 2);
do
    uv run python ../../../fire360/explanations/compute_explanations.py --dataset_name adult --model_name adult_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/fire360/artifacts/adult/synthetic_data/synthetic_data_100000_epochs_2500_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/fire360/artifacts/adult/bb/adult_BB.pth  --debug True --num_processes 20 --explanation_type knn --validation_seed $i --store_path /home/lcorbucci/fire360/artifacts/adult/explanations
done
