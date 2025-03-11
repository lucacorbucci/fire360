
for i in $(seq 11 12);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name letter --bb_path /home/lcorbucci/synth_xai/artifacts/letter/bb/letter_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/letter/comparison_explanation/lore/ --validation_seed $i
done