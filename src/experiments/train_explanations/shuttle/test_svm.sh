
# for i in $(seq 1 2);
# do
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name shuttle --model_name shuttle_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/shuttle/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth  --debug True --num_processes 20 --explanation_type svm --validation_seed $i --store_path /home/lcorbucci/synth_xai/artifacts/shuttle/explanations
# done


for i in $(seq 50 51);
do
    uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name shuttle --model_name shuttle_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/shuttle/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth  --debug True --num_processes 20 --explanation_type svm --validation_seed $i --store_path /home/lcorbucci/synth_xai/artifacts/shuttle/explanations
done


for i in $(seq 52 53);
do
    uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name shuttle --model_name shuttle_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/shuttle/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth  --debug True --num_processes 20 --explanation_type svm --validation_seed $i --store_path /home/lcorbucci/synth_xai/artifacts/shuttle/explanations
done
