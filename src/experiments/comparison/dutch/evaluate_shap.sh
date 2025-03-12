
uv run python ../../../synth_xai/explanations/evaluate_explanations.py --dataset_name dutch --bb_path /home/lcorbucci/synth_xai/artifacts/dutch/bb/dutch_BB.pth --explanation_type shap --explanations shap_11.pkl shap_12.pkl --artifacts_path /home/lcorbucci/synth_xai/artifacts/dutch/comparison_explanation/shap/ --top_k 3 5 8 10 20

