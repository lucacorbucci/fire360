
# for i in $(seq 1 2);
# do
#     uv run python ../../../fire360/explanations/compute_explanations.py --dataset_name covertype --model_name covertype_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/fire360/artifacts/covertype/synthetic_data/synthetic_data_200000_epochs_1000_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/fire360/artifacts/covertype/bb/covertype_BB.pth  --debug True --num_processes 20 --explanation_type svm --validation_seed $i --store_path /home/lcorbucci/fire360/artifacts/covertype/explanations
# done

for i in $(seq 50 51);
do
    uv run python ../../../fire360/explanations/compute_explanations.py --dataset_name covertype --model_name covertype_BB --top_k 2500 --synthetic_dataset_path /home/lcorbucci/fire360/artifacts/covertype/synthetic_data/synthetic_data_200000_epochs_1000_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/fire360/artifacts/covertype/bb/covertype_BB.pth  --debug True --num_processes 20 --explanation_type svm --validation_seed $i --store_path /home/lcorbucci/fire360/artifacts/covertype/explanations
done

for i in $(seq 52 53);
do
    uv run python ../../../fire360/explanations/compute_explanations.py --dataset_name covertype --model_name covertype_BB --top_k 5000 --synthetic_dataset_path /home/lcorbucci/fire360/artifacts/covertype/synthetic_data/synthetic_data_200000_epochs_1000_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/fire360/artifacts/covertype/bb/covertype_BB.pth  --debug True --num_processes 20 --explanation_type svm --validation_seed $i --store_path /home/lcorbucci/fire360/artifacts/covertype/explanations
done
