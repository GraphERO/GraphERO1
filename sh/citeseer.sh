CUDA_VISIBLE_DEVICES_DEFAULT=0

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_DEFAULT python main.py --embedder graphero --propagate True --dataset Citeseer --setting lt --imbalance_ratio 100 --noise True --noise_type uniform --noise_rate 0.2
