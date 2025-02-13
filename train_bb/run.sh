# for i in $(seq 0 3);
# do
#     uv run python /home/lcorbucci/personalized_explanations/train_bb/shuttle/../../train_bb.py --shuffle_seed $i --batch_size=61 --lr=0.023620400262700368 --optimizer=adam --epochs 10  --project_name Tango_BB --dataset_name shuttle
# done


# for i in $(seq 0 3);
# do
#     uv run python /home/lcorbucci/personalized_explanations/train_bb/letter/../../train_bb.py --shuffle_seed $i --batch_size=46 --lr=0.01595998943629738 --optimizer=adam --epochs 10  --project_name Tango_BB --dataset_name letter
# done

# for i in $(seq 0 3);
# do
#     uv run python /home/lcorbucci/personalized_explanations/train_bb/covertype/../../train_bb.py --shuffle_seed $i --batch_size=52 --lr=0.09256954362659872 --optimizer=sgd --epochs 10  --project_name Tango_BB --dataset_name covertype
# done

# for i in $(seq 0 3);
# do
#     uv run python /home/lcorbucci/personalized_explanations/train_bb/house16/../../train_bb.py --shuffle_seed $i --batch_size=49 --lr=0.028977425730221253 --optimizer=adam --epochs 10  --project_name Tango_BB --dataset_name house16
# done

# for i in $(seq 0 3);
# do
#     uv run python /home/lcorbucci/personalized_explanations/train_bb/dutch/../../train_bb.py --shuffle_seed $i --batch_size=50 --lr=0.009380641914751766 --optimizer=adam --epochs 10  --project_name Tango_BB --dataset_name dutch
# done

for i in $(seq 0 3);
do
    uv run python /home/lcorbucci/personalized_explanations/train_bb/adult/../../train_bb.py --shuffle_seed $i --batch_size=59 --lr=0.03706758980843329 --optimizer=adam --epochs 10  --project_name Tango_BB --dataset_name adult
done