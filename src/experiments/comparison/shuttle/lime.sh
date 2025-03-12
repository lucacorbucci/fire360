
# for i in $(seq 5 7);
# do
#     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name shuttle --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth --explanation_type lime --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/shuttle/comparison_explanation/lime/ --validation_seed $i
# done


for i in $(seq 50 51);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name shuttle --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth --explanation_type lime --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/shuttle/comparison_explanation/lime/ --validation_seed $i  --neigh_size 1000
done


for i in $(seq 52 53);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name shuttle --bb_path /home/lcorbucci/synth_xai/artifacts/shuttle/bb/shuttle_BB.pth --explanation_type lime --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/shuttle/comparison_explanation/lime/ --validation_seed $i  --neigh_size 2500
done