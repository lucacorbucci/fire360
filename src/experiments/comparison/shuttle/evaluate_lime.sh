
uv run python ../../../synth_xai/explanations/evaluate_explanations.py --dataset_name shuttle --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth --explanation_type lime --explanations lime_5.pkl lime_6.pkl --artifacts_path /home/lcorbucci/synth_xai/artifacts/shuttle/comparison_explanation/lime/ --top_k 3 5 8 10 20
