 
uv run python ../../../fire360/explanations/evaluate_explanations.py --dataset_name dutch --bb_path /home/lcorbucci/fire360/artifacts/dutch/bb/dutch_BB.pth --explanation_type shap --explanations shap_11.pkl shap_12.pkl --artifacts_path /home/lcorbucci/fire360/artifacts/dutch/comparison_explanation/shap/ --top_k 3 5 8 10 15 20 --wandb_project_name new_metrics_computation

