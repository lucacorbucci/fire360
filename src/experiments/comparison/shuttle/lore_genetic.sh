
for i in $(seq 5 6);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name shuttle --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/shuttle/comparison_explanation/lore/ --validation_seed $i --lore_generator genetic
done