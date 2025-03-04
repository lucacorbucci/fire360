
for i in $(seq 3 3);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name dutch --bb_path /home/lcorbucci/synth_xai/artifacts/dutch/bb/dutch_BB.pth --explanation_type lime --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/dutch/comparison_explanation/lime/ --validation_seed $i
done
