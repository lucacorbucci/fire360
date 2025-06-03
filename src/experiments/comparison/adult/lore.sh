
# for i in $(seq 11 12);
# do
#     uv run python ../../../fire360/comparison/compute_explanations.py --dataset_name adult --bb_path /home/lcorbucci/fire360/artifacts/adult/bb/adult_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/fire360/artifacts/adult/comparison_explanation/lore/ --validation_seed $i
# done

for i in $(seq 50 51);
do
    uv run python ../../../fire360/comparison/compute_explanations.py --dataset_name adult --bb_path /home/lcorbucci/fire360/artifacts/adult/bb/adult_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/fire360/artifacts/adult/comparison_explanation/lore/ --validation_seed $i  --neigh_size 1000
done

for i in $(seq 52 53);
do
    uv run python ../../../fire360/comparison/compute_explanations.py --dataset_name adult --bb_path /home/lcorbucci/fire360/artifacts/adult/bb/adult_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/fire360/artifacts/adult/comparison_explanation/lore/ --validation_seed $i --neigh_size 2500
done