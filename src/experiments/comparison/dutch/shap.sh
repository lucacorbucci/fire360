for i in $(seq 4 4);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name dutch --bb_path /home/lcorbucci/synth_xai/artifacts/dutch/bb/dutch_BB.pth --explanation_type shap --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/dutch/comparison_explanation/shap/  --validation_seed  $i
done
