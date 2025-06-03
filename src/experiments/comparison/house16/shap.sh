for i in $(seq 11 12);
do
    uv run python ../../../fire360/comparison/compute_explanations.py --dataset_name house16 --bb_path /home/lcorbucci/fire360/artifacts/house16/bb/house16_BB.pth --explanation_type shap --num_processes 20 --store_path /home/lcorbucci/fire360/artifacts/house16/comparison_explanation/shap/  --validation_seed  $i
done
