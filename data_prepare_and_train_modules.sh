python preprocess/train_filter_prepare.py &&
python train.py --gpu 0 --epochs 100 --batch_size 4 --lr 2e-5 --is_train True --is_load False --component filter &&
python preprocess/train_rank_prepare.py &&
python train.py --gpu 0 --epochs 100 --batch_size 4 --lr 2e-5 --is_train True --is_load False --component rank