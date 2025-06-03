for i in $(seq 11 12);
do
    uv run python ../../../fire360/comparison/compute_explanations.py --dataset_name letter --bb_path /home/lcorbucci/fire360/artifacts/letter/bb/letter_BB.pth --explanation_type shap --num_processes 20 --store_path /home/lcorbucci/fire360/artifacts/letter/comparison_explanation/shap/  --validation_seed  $i --k_means_k 200
done
