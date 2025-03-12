
# for i in $(seq 11 12);
# do
#     uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name covertype --bb_path /home/lcorbucci/synth_xai/artifacts/covertype/bb/covertype_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/covertype/comparison_explanation/lore/ --validation_seed $i --lore_generator genetic --num_explained_instances 10000
# done

for i in $(seq 50 51);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name covertype --bb_path /home/lcorbucci/synth_xai/artifacts/covertype/bb/covertype_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/covertype/comparison_explanation/lore/ --validation_seed $i --lore_generator genetic --num_explained_instances 10000  --neigh_size 1000
done

for i in $(seq 52 53);
do
    uv run python ../../../synth_xai/comparison/compute_explanations.py --dataset_name covertype --bb_path /home/lcorbucci/synth_xai/artifacts/covertype/bb/covertype_BB.pth --explanation_type lore --num_processes 20 --store_path /home/lcorbucci/synth_xai/artifacts/covertype/comparison_explanation/lore/ --validation_seed $i --lore_generator genetic --num_explained_instances 10000  --neigh_size 2500
done