# Adult
# DT
    uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name adult --model_name adult_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/adult/synthetic_data/synthetic_data_100000_epochs_2500_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/synth_xai/artifacts/adult/bb/adult_BB.pth  --debug True --num_processes 1 --explanation_type dt --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/adult/explanations

# SVM
    uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name adult --model_name adult_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/adult/synthetic_data/synthetic_data_100000_epochs_2500_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/synth_xai/artifacts/adult/bb/adult_BB.pth  --debug True --num_processes 1 --explanation_type svm --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/adult/explanations

# LR
    uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name adult --model_name adult_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/adult/synthetic_data/synthetic_data_100000_epochs_2500_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/synth_xai/artifacts/adult/bb/adult_BB.pth  --debug True --num_processes 1 --explanation_type logistic --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/adult/explanations

# KNN
    uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name adult --model_name adult_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/adult/synthetic_data/synthetic_data_100000_epochs_2500_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/synth_xai/artifacts/adult/bb/adult_BB.pth  --debug True --num_processes 1 --explanation_type knn --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/adult/explanations
# Lime
uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name adult --bb_path /home/lcorbucci/synth_xai/artifacts/adult/bb/adult_BB.pth --explanation_type lime --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/adult/comparison_explanation/lime/ --validation_seed 1 --project_name time_computation --num_explained_instances 100
# # Shap
uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name adult --bb_path /home/lcorbucci/synth_xai/artifacts/adult/bb/adult_BB.pth --explanation_type shap --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/adult/comparison_explanation/shap/  --validation_seed 1 --project_name time_computation --num_explained_instances 100
# # Lore
uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name adult --bb_path /home/lcorbucci/synth_xai/artifacts/adult/bb/adult_BB.pth --explanation_type lore --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/adult/comparison_explanation/lore/ --validation_seed 1 --project_name time_computation --num_explained_instances 100


# # Dutch 
# # Lime
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name dutch --bb_path /home/lcorbucci/synth_xai/artifacts/dutch/bb/dutch_BB.pth --explanation_type lime --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/dutch/comparison_explanation/lime/ --validation_seed 1 --project_name time_computation --num_explained_instances 100
# # # Shap
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name dutch --bb_path /home/lcorbucci/synth_xai/artifacts/dutch/bb/dutch_BB.pth --explanation_type shap --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/dutch/comparison_explanation/shap/  --validation_seed 1 --project_name time_computation --num_explained_instances 100

# # # Lore
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name dutch --bb_path /home/lcorbucci/synth_xai/artifacts/dutch/bb/dutch_BB.pth --explanation_type lore --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/dutch/comparison_explanation/lore/ --validation_seed 1 --project_name time_computation --num_explained_instances 100

# # DT
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name dutch --model_name dutch_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/dutch/synthetic_data/synthetic_data_100000_epochs_2500_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/synth_xai/artifacts/dutch/bb/dutch_BB.pth  --debug True --num_processes 1 --explanation_type dt --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/dutch/explanations
# # SVM
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name dutch --model_name dutch_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/dutch/synthetic_data/synthetic_data_100000_epochs_2500_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/synth_xai/artifacts/dutch/bb/dutch_BB.pth  --debug True --num_processes 1 --explanation_type svm --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/dutch/explanations
# # LR
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name dutch --model_name dutch_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/dutch/synthetic_data/synthetic_data_100000_epochs_2500_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/synth_xai/artifacts/dutch/bb/dutch_BB.pth  --debug True --num_processes 1 --explanation_type logistic --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/dutch/explanations
# # KNN
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name dutch --model_name dutch_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/dutch/synthetic_data/synthetic_data_100000_epochs_2500_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/synth_xai/artifacts/dutch/bb/dutch_BB.pth  --debug True --num_processes 1 --explanation_type knn --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/dutch/explanations


# # Letter
# # Lime
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name letter --bb_path /home/lcorbucci/synth_xai/artifacts/letter/bb/letter_BB.pth --explanation_type lime --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/letter/comparison_explanation/lime/ --validation_seed 1 --project_name time_computation --num_explained_instances 100

# # # Shap
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name letter --bb_path /home/lcorbucci/synth_xai/artifacts/letter/bb/letter_BB.pth --explanation_type shap --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/letter/comparison_explanation/shap/  --validation_seed 1 --project_name time_computation --num_explained_instances 100 --k_means_k 200

# # # Lore
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name letter --bb_path /home/lcorbucci/synth_xai/artifacts/letter/bb/letter_BB.pth --explanation_type lore --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/letter/comparison_explanation/lore/ --validation_seed 1 --project_name time_computation --num_explained_instances 100

# # DT
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name letter --model_name letter_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/letter/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/letter/bb/letter_BB.pth  --debug True --num_processes 1 --explanation_type dt --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/letter/explanations
# # SVM
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name letter --model_name letter_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/letter/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/letter/bb/letter_BB.pth  --debug True --num_processes 1 --explanation_type svm --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/letter/explanations
# # LR
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name letter --model_name letter_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/letter/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/letter/bb/letter_BB.pth  --debug True --num_processes 1 --explanation_type logistic --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/letter/explanations
# # KNN
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name letter --model_name letter_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/letter/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/letter/bb/letter_BB.pth  --debug True --num_processes 1 --explanation_type knn --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/letter/explanations


# # Shuttle
# # Lime
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name shuttle --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth --explanation_type lime --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/shuttle/comparison_explanation/lime/ --validation_seed 1 --project_name time_computation --num_explained_instances 100

# # # Shap
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name shuttle --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth --explanation_type shap --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/shuttle/comparison_explanation/shap/  --validation_seed 1 --project_name time_computation --num_explained_instances 100

# # # Lore
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name shuttle --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth --explanation_type lore --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/shuttle/comparison_explanation/lore/ --validation_seed 1 --project_name time_computation --num_explained_instances 100

# # DT
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name shuttle --model_name shuttle_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/shuttle/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth  --debug True --num_processes 1 --explanation_type dt --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/shuttle/explanations
# # SVM
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name shuttle --model_name shuttle_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/shuttle/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth  --debug True --num_processes 1 --explanation_type svm --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/shuttle/explanations
# # LR
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name shuttle --model_name shuttle_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/shuttle/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth  --debug True --num_processes 1 --explanation_type logistic --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/shuttle/explanations
# # KNN
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name shuttle --model_name shuttle_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/shuttle/synthetic_data/synthetic_data_200000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth  --debug True --num_processes 1 --explanation_type knn --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/shuttle/explanations


# # House16
# # Lime
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name house16 --bb_path /home/lcorbucci/synth_xai/artifacts/house16/bb/house16_BB.pth --explanation_type lime --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/house16/comparison_explanation/lime/ --validation_seed 1 --project_name time_computation --num_explained_instances 100

# # # Shap
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name house16 --bb_path /home/lcorbucci/synth_xai/artifacts/house16/bb/house16_BB.pth --explanation_type shap --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/house16/comparison_explanation/shap/  --validation_seed 1 --project_name time_computation --num_explained_instances 100

# # # Lore
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name house16 --bb_path /home/lcorbucci/synth_xai/artifacts/house16/bb/house16_BB.pth --explanation_type lore --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/house16/comparison_explanation/lore/ --validation_seed 1 --project_name time_computation --num_explained_instances 100

# # DT
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name house16 --model_name house16_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/house16/synthetic_data/synthetic_data_150000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/house16/bb/house16_BB.pth  --debug True --num_processes 1 --explanation_type dt --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/house16/explanations
# # SVM
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name house16 --model_name house16_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/house16/synthetic_data/synthetic_data_150000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/house16/bb/house16_BB.pth  --debug True --num_processes 1 --explanation_type svm --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/house16/explanations
# # LR
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name house16 --model_name house16_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/house16/synthetic_data/synthetic_data_150000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/house16/bb/house16_BB.pth  --debug True --num_processes 1 --explanation_type logistic --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/house16/explanations
# # KNN
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name house16 --model_name house16_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/house16/synthetic_data/synthetic_data_150000_epochs_2500_synthethizer_name_ctgan.csv --bb_path /home/lcorbucci/synth_xai/artifacts/house16/bb/house16_BB.pth  --debug True --num_processes 1 --explanation_type knn --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/house16/explanations


# # Covertype
# # Lime
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name covertype --bb_path /home/lcorbucci/synth_xai/artifacts/covertype/bb/covertype_BB.pth --explanation_type lime --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/covertype/comparison_explanation/lime/ --validation_seed 1 --project_name time_computation --num_explained_instances 100

# # # Shap
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name covertype --bb_path /home/lcorbucci/synth_xai/artifacts/covertype/bb/covertype_BB.pth --explanation_type shap --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/covertype/comparison_explanation/shap/  --validation_seed 1 --project_name time_computation --num_explained_instances 100

# # # Lore
# #     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name covertype --bb_path /home/lcorbucci/synth_xai/artifacts/covertype/bb/covertype_BB.pth --explanation_type lore --num_processes 1 --store_path /home/lcorbucci/synth_xai/artifacts/covertype/comparison_explanation/lore/ --validation_seed 1 --project_name time_computation --num_explained_instances 100

# # DT
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name covertype --model_name covertype_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/covertype/synthetic_data/synthetic_data_200000_epochs_1000_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/synth_xai/artifacts/covertype/bb/covertype_BB.pth  --debug True --num_processes 1 --explanation_type dt --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/covertype/explanations
# # SVM
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name covertype --model_name covertype_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/covertype/synthetic_data/synthetic_data_200000_epochs_1000_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/synth_xai/artifacts/covertype/bb/covertype_BB.pth  --debug True --num_processes 1 --explanation_type svm --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/covertype/explanations
# # LR
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name covertype --model_name covertype_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/covertype/synthetic_data/synthetic_data_200000_epochs_1000_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/synth_xai/artifacts/covertype/bb/covertype_BB.pth  --debug True --num_processes 1 --explanation_type logistic --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/covertype/explanations
# # KNN
#     uv run python ../../../synth_xai/explanations/compute_explanations.py --dataset_name covertype --model_name covertype_BB --top_k 1000 --synthetic_dataset_path /home/lcorbucci/synth_xai/artifacts/covertype/synthetic_data/synthetic_data_200000_epochs_1000_synthethizer_name_tvae.csv --bb_path /home/lcorbucci/synth_xai/artifacts/covertype/bb/covertype_BB.pth  --debug True --num_processes 1 --explanation_type knn --validation_seed 1 --project_name time_computation --num_explained_instances 100 --store_path /home/lcorbucci/synth_xai/artifacts/covertype/explanations
