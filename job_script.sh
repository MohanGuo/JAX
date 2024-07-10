#!/bin/bash

export WANDB_API_KEY=71a1e9fee0054f7f82f52b84353b9eb2d0430489

python /home/mguo/JAX/e3_diffusion_for_molecules-main_jax/main_qm9.py --exp_name edm_qm9  --model egnn_dynamics --lr 1e-4  --nf 256 --n_layers 9 --diffusion_steps 100 --sin_embedding False --n_epochs 100 --n_stability_samples 200 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --diffusion_loss_type l2 --batch_size 64 --normalize_factors [1,4,10] --clip_grad False --conditioning alpha --dataset qm9_second_half --test_epochs 2 --exp_name jax_imp

