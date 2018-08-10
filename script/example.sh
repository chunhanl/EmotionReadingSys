cd ..

python train.py --dir emov1  --fc_secondlast 4096 --fc_last 4096  --epoch 200 --save_period 20
python train.py --dir emov2  --fc_secondlast 2048 --fc_last 0     --epoch 200 --save_period 20



