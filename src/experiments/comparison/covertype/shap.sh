for i in $(seq 1 2);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name covertype --bb_path /home/lcorbucci/synth_xai/artifacts/covertype/bb/covertype_BB.pth --explanation_type shap --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/covertype/comparison_explanation/shap/  --validation_seed 1
done
