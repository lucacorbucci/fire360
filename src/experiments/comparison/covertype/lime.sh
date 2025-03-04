
for i in $(seq 1 3);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name covertype --bb_path /home/lcorbucci/synth_xai/artifacts/covertype/bb/covertype_BB.pth --explanation_type lime --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/covertype/comparison_explanation/lime/ --validation_seed $i
done