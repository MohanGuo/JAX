# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model, DistributionNodes, DistributionProperty
from equivariant_diffusion import luke_en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, prepare_context_torch, compute_mean_mad #get_checkpoint_info
from train_test import train_epoch, test, analyze_and_save
import jax
from flax.training import train_state
import optax
import os
import flax

from equivariant_diffusion.utils import assert_mean_zero_with_mask, assert_mean_zero_with_mask_torch, remove_mean_with_mask, remove_mean_with_mask_torch, \
    assert_correctly_masked, assert_correctly_masked_torch, sample_center_gravity_zero_gaussian_with_mask
from qm9.analyze import analyze_stability_for_molecules
import qm9.utils as qm9utils
import jax.numpy as jnp
import numpy as np

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv')
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.99999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
args = parser.parse_args()

dataset_info = get_dataset_info(args.dataset, args.remove_h)

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
gpu_present = any(device.platform == 'gpu' for device in jax.devices())
device = "cuda" if gpu_present else "cpu"
args.run_trace=False

dtype = torch.float32
# print(device)
if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(args)

utils.create_folders(args)
# print(args)


# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

data_dummy = next(iter(dataloaders['train']))

if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.shape[2]
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf



gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

def check_mask_correct_torch(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked_torch(variable, node_mask)


def count_params(params):
    total_params = 0
    for key, value in params.items():
        if isinstance(value, dict):
            total_params += count_params(value)
        else:
            total_params += value.size
    return total_params


def save_model_params_and_count_to_file(params, file_path):
    # params = model.params  # Assuming 'model' is a Flax model instance with loaded params
    with open(file_path, 'w') as f:
        for name, param in flax.traverse_util.flatten_dict(params).items():
            param_count = param.size
            f.write(f"Parameter name: {name}\n")
            f.write(f"Parameter count: {param_count}\n")
            f.write(f"Parameter values: {param}\n\n")


def test_stability():
    # seed = 42
    # rng = jax.random.PRNGKey(seed)
    # rng, rng_init = jax.random.split(rng, 2)

    # params, model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'], rng_init,
    #                                                  property_norms)
    
    histogram = dataset_info['n_nodes']
    dataloader_train = dataloaders['train']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)
    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)
        # prop_dist = DistributionPropertyJax(JAXDataLoaderWrapper(dataloader_train), args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    # # Prepare data in torch
    # dtype = torch.float32
    # exmp_imgs = next(iter(dataloader_train))
    # x = exmp_imgs['positions'].to(device, dtype)
    # node_mask = exmp_imgs['atom_mask'].to(device, dtype).unsqueeze(2)
    # edge_mask = exmp_imgs['edge_mask'].to(device, dtype)
    # one_hot = exmp_imgs['one_hot'].to(device, dtype)
    # charges = (exmp_imgs['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

    # x = remove_mean_with_mask_torch(x, node_mask)

    # if args.augment_noise > 0:
    #     # Add noise eps ~ N(0, augment_noise) around points.
    #     eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
    #     x = x + eps * args.augment_noise

    # x = remove_mean_with_mask_torch(x, node_mask)
    # if args.data_augmentation:
    #     x = utils.random_rotation(x).detach()

    # check_mask_correct_torch([x, one_hot, charges], node_mask)
    # assert_mean_zero_with_mask_torch(x, node_mask)

    # h = {'categorical': one_hot, 'integer': charges}

    # if len(args.conditioning) > 0:
    #     context = qm9utils.prepare_context_torch(args.conditioning, exmp_imgs, property_norms).to(device, dtype)
    #     assert_correctly_masked_torch(context, node_mask)
    # else:
    #     context = None
    # # print(f"Variables in test_stability: {one_hot.shape}, {one_hot}, {x.shape}, {node_mask.shape}")

    # molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    # molecules['one_hot'].append(one_hot)
    # molecules['x'].append(x)
    # molecules['node_mask'].append(node_mask)


    #Prepare data in JAX
    dtype = jnp.float32
    exmp_imgs = next(iter(dataloader_train))
    x_example = exmp_imgs['positions'].astype(dtype)
    node_mask_example = jnp.expand_dims(exmp_imgs['atom_mask'],2).astype(dtype)
    edge_mask_example = exmp_imgs['edge_mask'].astype(dtype)
    one_hot_example = exmp_imgs['one_hot'].astype(dtype)
    charges_example = (exmp_imgs['charges'] if args.include_charges else jnp.zeros(0)).astype(dtype)
    h_example = {'categorical': one_hot_example, 'integer': charges_example}

    x_example = remove_mean_with_mask(x_example, node_mask_example)
    if args.augment_noise > 0:
        # Add noise eps ~ N(0, augment_noise) around points.
        eps = sample_center_gravity_zero_gaussian_with_mask(x_example.size(), x_example.device, node_mask_example)
        x_example = x_example + eps * args.augment_noise

    x_example = remove_mean_with_mask(x_example, node_mask_example)
    if args.data_augmentation:
        x_example = utils.random_rotation(x_example).detach()

    check_mask_correct([x_example, one_hot_example, charges_example], node_mask_example)
    assert_mean_zero_with_mask(x_example, node_mask_example)

    # print(f"h: {h_example}")
    if len(args.conditioning) > 0:
        context_example = qm9utils.prepare_context(args.conditioning, exmp_imgs, property_norms)
        # context_cpu = context_example.data.cpu()
        # context_example = jnp.asarray(context_cpu)
        assert_correctly_masked(context_example, node_mask_example)
    else:
        context_example = None

    one_hot_np = np.array(jax.device_get(one_hot_example))
    x_np = np.array(jax.device_get(x_example))
    node_mask_np = np.array(jax.device_get(node_mask_example))

    one_hot_torch = torch.from_numpy(one_hot_np)
    x_torch = torch.from_numpy(x_np)
    node_mask_torch = torch.from_numpy(node_mask_np)
    print(f"Variables in test_stability: {one_hot_torch.shape}, {one_hot_torch}, {x_torch.shape}, {node_mask_torch.shape}")

    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    molecules['one_hot'].append(one_hot_torch)
    molecules['x'].append(x_torch)
    molecules['node_mask'].append(node_mask_torch)

    #Analyze stability

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)
    print("In analyze_and_save, log stability!")
    # wandb.log(validity_dict)
    # if rdkit_tuple is not None:
    #     wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict

def main():

    # Initialize dataparallel if enabled and possible.
    # if args.dp and torch.cuda.device_count() > 1:
    #     print(f'Training using {torch.cuda.device_count()} GPUs')
    #     model_dp = torch.nn.DataParallel(model.cpu())
    #     model_dp = model_dp.cuda()
    # else:
    #     model_dp = model

    # Create EGNN flow
    seed = 42
    rng = jax.random.PRNGKey(seed)
    rng, rng_init = jax.random.split(rng, 2)

    params, model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'], rng_init,
                                                     property_norms)
    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)

    #Test model
    # file_path = 'parameters.txt'
    # if not os.path.exists(file_path):
    #     open(file_path, 'w').close()
    # with open(file_path, "a") as file:
    #     file.write(str(params) + "\n")
    total_params = count_params(params)
    print(f"Total number of parameters: {total_params}")
    save_model_params_and_count_to_file(params, "jax_model_parameters.txt")
    
   
    
    #Optimizer
    optim = optax.adamw(
        learning_rate=args.lr, 
        weight_decay=1e-12)

    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=params,
                                            tx=optim)
    
    model_dp = model_state

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        ema_state = copy.deepcopy(model_state)
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        # model_ema_dp = model_dp
    model_ema_dp = None
    

    
    # ckpt_mngr,ckpt_path=get_checkpoint_info(args)

    # if(args.resume is not None ):
    #     if(os.path.exists(ckpt_path)):
    #         print("ckpt_path is ",ckpt_path)
    #         step=ckpt_mngr.latest_step()
    #         model_state = ckpt_mngr.restore(step)
    #     else:
    #         print("could not find checkpoint at ",ckpt_path)

    #Train
    if(args.augment_noise>0):
        print("unfortunately we've broken augment noise")
    best_nll_val = 1e8
    best_nll_test = 1e8
    # for i,  data in enumerate(dataloaders['train']):
    #     print("i is ",i)


    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        # args.break_train_epoch = True
        params, model_state, average_loss, ema_state = train_epoch(
            rng=rng, args=args, loader=dataloaders['train'], epoch=epoch, model=model, params=model_state.params,
            model_dp=model_dp,
            model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
            nodes_dist=nodes_dist, dataset_info=dataset_info,
            gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist, state=model_state, ema_state=ema_state
        )
        average_loss.block_until_ready()
        print(f"-------  Epoch took {time.time() - start_epoch:.1f} seconds. Average Loss: {average_loss}")

        if epoch % args.test_epochs == 0:
            if isinstance(model, luke_en_diffusion.EnVariationalDiffusion):
                wandb.log(model_state.apply_fn(model_state.params, mode="log_info"), commit=True)

            if not args.break_train_epoch:
                analyze_and_save(rng=rng, args=args, epoch=epoch, model_sample=ema_state, nodes_dist=nodes_dist,
                                 dataset_info=dataset_info, device=device,
                                 prop_dist=prop_dist, n_samples=args.n_stability_samples)
                print(f"-------  analyze_and_save took {time.time() - start_epoch:.1f} seconds.")

            nll_val = test(rng, args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms, model_state=ema_state)
            nll_test = test(rng, args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms, model_state=ema_state)

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test

            print(f"-------  test took {time.time() - start_epoch:.1f} seconds.")

            print('Val loss: %.4f \t Test loss:  %.4fBest val loss: %.4f \t Best test loss:  %.4f'  % (nll_val, nll_test, best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)



if __name__ == "__main__":
    main()
    # result = test_stability()
    # print(result)