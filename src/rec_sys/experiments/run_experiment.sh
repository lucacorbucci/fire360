# # Adult
# uv run python rec_sys.py --batch_size 251 --dropout 0.17915805355084274 --hidden_size 186 --lr 0.0023275428445226265 --num_epochs 142 --train_path ../data/train_adult_simpler.csv --test_path ../data/test_adult_simpler.csv --include_previous_targets True --project_name ResultsRS

# # Dutch
# uv run python rec_sys.py --batch_size 113 --dropout 0.2496565300297116 --hidden_size 121 --lr 0.0013749972758768676 --num_epochs 147 --train_path ../data/train_dutch_simpler.csv --test_path ../data/test_dutch_simpler.csv --include_previous_targets True --project_name ResultsRS

# shuttle
uv run python rec_sys.py --batch_size 244 --dropout 0.2311799666366024 --hidden_size 320 --lr 0.013920799624477044 --num_epochs 32 --train_path ../data/train_shuttle.csv --test_path ../data/test_shuttle.csv --include_previous_targets True --project_name ResultsRS

# house16
uv run python rec_sys.py --batch_size 246 --dropout 0.24977802637155108 --hidden_size 109 --lr 0.009400904159879934 --num_epochs 88 --train_path ../data/train_house16.csv --test_path ../data/test_house16.csv --include_previous_targets True --project_name ResultsRS

# letter
uv run python rec_sys.py --batch_size 210 --dropout 0.28623411028369866 --hidden_size 316 --lr 0.0514603023242737 --num_epochs 115 --train_path ../data/train_letter.csv --test_path ../data/test_letter.csv --include_previous_targets True --project_name ResultsRS

# # covtype
uv run python rec_sys.py --batch_size 146 --dropout 0.203229777115021 --hidden_size 106 --lr 0.0514603023242737 --num_epochs 142 --train_path ../data/train_covertype.csv --test_path ../data/test_covertype.csv --include_previous_targets True --project_name ResultsRS
