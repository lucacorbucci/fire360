
for i in $(seq 1 2);
do
    uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name letter --model_name letter_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/letter/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/letter/bb/letter_BB.pth  --debug True --num_processes 20 --explanation_type logistic --validation_seed $i --store_path /home/lcorbucci/synth_xai/artifacts/letter/explanations
done
