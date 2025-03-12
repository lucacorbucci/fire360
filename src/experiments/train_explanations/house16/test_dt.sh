
# for i in $(seq 1 2);
# do
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name house16 --model_name house16_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/house16/synthetic_data/synthetic_data_150000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/house16/bb/house16_BB.pth  --debug True --num_processes 20 --explanation_type dt --validation_seed $i --store_path /home/lcorbucci/synth_xai/artifacts/house16/explanations
# done

for i in $(seq 50 51);
do
    uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name house16 --model_name house16_BB --top_k 2500 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/house16/synthetic_data/synthetic_data_150000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/house16/bb/house16_BB.pth  --debug True --num_processes 20 --explanation_type dt --validation_seed $i --store_path /home/lcorbucci/synth_xai/artifacts/house16/explanations
done

for i in $(seq 51 52);
do
    uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name house16 --model_name house16_BB --top_k 5000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/house16/synthetic_data/synthetic_data_150000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/house16/bb/house16_BB.pth  --debug True --num_processes 20 --explanation_type dt --validation_seed $i --store_path /home/lcorbucci/synth_xai/artifacts/house16/explanations
done