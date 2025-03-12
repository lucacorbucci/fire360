
# for i in $(seq 11 12);
# do
#     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name letter --bb_path /home/lcorbucci/synth_xai/artifacts/letter/bb/letter_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/letter/comparison_explanation/lore/ --validation_seed $i --lore_generator genetic
# done


for i in $(seq 50 51);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name letter --bb_path /home/lcorbucci/synth_xai/artifacts/letter/bb/letter_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/letter/comparison_explanation/lore/ --validation_seed $i --lore_generator genetic  --neigh_size 1000
done


for i in $(seq 52 53);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name letter --bb_path /home/lcorbucci/synth_xai/artifacts/letter/bb/letter_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/letter/comparison_explanation/lore/ --validation_seed $i --lore_generator genetic  --neigh_size 2500
done