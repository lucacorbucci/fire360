for i in $(seq 2 2);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name letter --bb_path /home/lcorbucci/synth_xai/artifacts/letter/bb/letter_BB.pth --explanation_type shap --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/letter/comparison_explanation/shap/  --validation_seed  $i
done
