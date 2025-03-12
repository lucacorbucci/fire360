
# for i in $(seq 11 12);
# do
#     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name house16 --bb_path /home/lcorbucci/synth_xai/artifacts/house16/bb/house16_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/house16/comparison_explanation/lore/ --validation_seed $i
# done


for i in $(seq 50 51);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name house16 --bb_path /home/lcorbucci/synth_xai/artifacts/house16/bb/house16_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/house16/comparison_explanation/lore/ --validation_seed $i  --neigh_size 1000
done


for i in $(seq 52 53);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name house16 --bb_path /home/lcorbucci/synth_xai/artifacts/house16/bb/house16_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/house16/comparison_explanation/lore/ --validation_seed $i  --neigh_size 2500
done