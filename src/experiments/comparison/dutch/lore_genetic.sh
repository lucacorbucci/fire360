
# for i in $(seq 11 12);
# do
#     uv run python ../../../fire360/comparison/compute_explanations.py --dataset_name dutch --bb_path /home/lcorbucci/fire360/artifacts/dutch/bb/dutch_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/fire360/artifacts/dutch/comparison_explanation/lore/ --validation_seed $i --lore_generator genetic
# done


for i in $(seq 50 51);
do
    uv run python ../../../fire360/comparison/compute_explanations.py --dataset_name dutch --bb_path /home/lcorbucci/fire360/artifacts/dutch/bb/dutch_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/fire360/artifacts/dutch/comparison_explanation/lore/ --validation_seed $i --lore_generator genetic --neigh_size 1000
done


for i in $(seq 52 53);
do
    uv run python ../../../fire360/comparison/compute_explanations.py --dataset_name dutch --bb_path /home/lcorbucci/fire360/artifacts/dutch/bb/dutch_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/fire360/artifacts/dutch/comparison_explanation/lore/ --validation_seed $i --lore_generator genetic --neigh_size 2500
done