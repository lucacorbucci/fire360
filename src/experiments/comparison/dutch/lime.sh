
# for i in $(seq 5 7);
# do
#     uv run python ../../../fire360/comparison/compute_explanations.py --dataset_name dutch --bb_path /home/lcorbucci/fire360/artifacts/dutch/bb/dutch_BB.pth --explanation_type lime --num_processes 20 --store_path /home/lcorbucci/fire360/artifacts/dutch/comparison_explanation/lime/ --validation_seed $i
# done


for i in $(seq 50 51);
do
    uv run python ../../../fire360/comparison/compute_explanations.py --dataset_name dutch --bb_path /home/lcorbucci/fire360/artifacts/dutch/bb/dutch_BB.pth --explanation_type lime --num_processes 20 --store_path /home/lcorbucci/fire360/artifacts/dutch/comparison_explanation/lime/ --validation_seed $i  --neigh_size 1000
done


for i in $(seq 52 53);
do
    uv run python ../../../fire360/comparison/compute_explanations.py --dataset_name dutch --bb_path /home/lcorbucci/fire360/artifacts/dutch/bb/dutch_BB.pth --explanation_type lime --num_processes 20 --store_path /home/lcorbucci/fire360/artifacts/dutch/comparison_explanation/lime/ --validation_seed $i  --neigh_size 2500
done