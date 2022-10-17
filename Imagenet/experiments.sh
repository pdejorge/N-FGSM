# Experiments with Imagenet dataset 
# Code uses DataParallel so it is expected to work with > 2 GPU's 

## N-FGSM
# Epsilon 2
sh run_fast_all.sh 2 2.0 1.0 -1

# Epsilon 4
sh run_fast_all_short_warmup.sh 4 4.0 1.0 -1

# Epsilon 6
sh run_fast_all_short_warmup.sh 6 4.0 1.0 -1

## RS-FGSM
# Epsilon 2
sh run_fast_all.sh 2 2.0 1.0 1

# Epsilon 4
sh run_fast_all_short_warmup.sh 4 4.0 1.0 1

# Epsilon 6
sh run_fast_all_short_warmup.sh 6 6.0 1.0 1

## FGSM
# Epsilon 2
sh run_fast_all.sh 2 2.0 0.0 -1

# Epsilon 4
sh run_fast_all_short_warmup.sh 4 4.0 0.0 -1

# Epsilon 6
sh run_fast_all_short_warmup.sh 6 6.0 0.0 -1
