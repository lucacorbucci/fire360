
for i in $(seq 1 3);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name letter --bb_path /home/lcorbucci/synth_xai/artifacts/letter/bb/letter_BB.pth --explanation_type lime --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/letter/comparison_explanation/lime/ --validation_seed $i
done
