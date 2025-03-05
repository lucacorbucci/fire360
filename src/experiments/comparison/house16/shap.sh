for i in $(seq 1 2);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name house16 --bb_path /home/lcorbucci/synth_xai/artifacts/house16/bb/house16_BB.pth --explanation_type shap --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/house16/comparison_explanation/shap/  --validation_seed  $i
done
