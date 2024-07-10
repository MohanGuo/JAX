import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask, \
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import qm9.utils as qm9utils
from qm9 import losses
import time
import torch
import jax.numpy as jnp
import jax
from flax import linen as nn
# from flax.training import train_state, checkpoints
# import optax
import os


# import tensorflow.python.profiler.trace
import time

def loss_fn(rng, state, params, x, h, node_mask, edge_mask, context, log_pN, ode_regularization):
    h_int=h["integer"]
    h_cat=h["categorical"]
    nll, reg_term, mean_abs_z = losses.compute_loss_and_nll_train(rng, state, params, log_pN, x, h_int,h_cat, node_mask,edge_mask, context)


    loss = nll + ode_regularization * reg_term

    loss = loss.sum()
    return loss, (nll, reg_term, mean_abs_z)


def update_model_average(ma_params, current_params, beta):
    # return jax.tree.map(update_average, ma_params, current_params)
    return jax.tree.map(lambda old, new: update_average(old, new, beta), ma_params, current_params)

def update_average(old, new, beta):
    if old is None:
        return new
    return old * beta + (1 - beta) * new


@jax.jit
def train_step(rng, state, x, h, node_mask, edge_mask, context, log_pN, ode_regularization, ema_state):
    
    # Gradient functionf
    grad_fn = jax.value_and_grad(loss_fn,  # Function to calculate the loss
                                 argnums=2,  # Parameters are THIRD argument of the function
                                 has_aux=True,  # Function has additional outputs, here accuracy
                                 allow_int=True  #TODO else error raised
                                 )
    # Determine gradients for current model, parameters and batch
    (loss, (nll, reg_term, mean_abs_z)), grads = grad_fn(rng, state, state.params, x, h, node_mask, edge_mask, context, log_pN, ode_regularization)
    state = state.apply_gradients(grads=grads)
    # new_ema_params = None
    # new_ema_params = ema.update_model_average(ema_state.params, state.params)
    beta = 0
    new_ema_params = update_model_average(ema_state.params, state.params, beta)
    ema_state = ema_state.replace(params=new_ema_params)
    # Return state and any other value we might want
    return state, loss, nll, reg_term, mean_abs_z, ema_state

# train_step = jax.jit(train_step, static_argnames='ema')

#Purpose: pad inputs so they all have the same number of nodes
#otherwise, jit has to recompile, making it slower
def pad_inputs(x, node_mask, h_cat, h_int, context, edge_mask, max_nodes=29):
    
    #max 
    mask_pad=0   
    max_edges=max_nodes*max_nodes
    
    #padding amount
    bs, initial_n_nodes, _ = x.shape
    pad_dims=max_nodes-initial_n_nodes
    pad_tuple=(0,pad_dims)
    assert(max_edges>=initial_n_nodes)
    assert(bs==64)

    x=jnp.pad(x,((0,0),pad_tuple,(0,0)),'constant', constant_values=(0))
    node_mask=jnp.pad(node_mask,((0,0),pad_tuple,(0,0)),'constant', constant_values=(mask_pad))
    h_cat=jnp.pad(h_cat,((0,0),pad_tuple,(0,0)),'constant', constant_values=(0))
    h_int=jnp.pad(h_int,((0,0),pad_tuple,(0,0)),'constant', constant_values=(0))
    context=jnp.pad(context,((0,0),pad_tuple,(0,0)),'constant', constant_values=(0))
    reshaped_edge_mask=jnp.reshape(edge_mask,(bs, initial_n_nodes,initial_n_nodes))
    edge_mask=jnp.pad(reshaped_edge_mask,((0,0),pad_tuple,pad_tuple),'constant', constant_values=(mask_pad)).reshape(-1,1)


    return x, node_mask, h_cat, h_int, context, edge_mask

def train_epoch(rng, args, loader, epoch, model, params, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist, state, ema_state):

    nll_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        dtype = jnp.float32
        x = data['positions'].astype(dtype)
        node_mask = jnp.expand_dims(data['atom_mask'],2).astype(dtype)
        edge_mask = data['edge_mask'].astype(dtype)
        one_hot = data['one_hot'].astype(dtype)
        charges = (data['charges'] if args.include_charges else jnp.zeros(0)).astype(dtype)

        x = remove_mean_with_mask(x, node_mask)
        # Add noise eps ~ N(0, augment_noise) around points.
        # eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
        # x = x + eps * args.augment_noise
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        max_value = check_mask_correct([x, one_hot, charges], node_mask)
        assert max_value < 1e-4, "The maximum absolute value must be less than 1e-4"

        assert_mean_zero_with_mask(x, node_mask)


        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms)
            # context_cpu = context.data.cpu()
            # context = jnp.asarray(context_cpu)
            assert_correctly_masked(context, node_mask)
        else:
            context = None


        N = node_mask.squeeze(2).sum(1).astype(jnp.float32)
        log_pN = nodes_dist.log_prob(N)
        x, node_mask, one_hot, charges, context, edge_mask = pad_inputs(x, node_mask, one_hot, charges, context, edge_mask, max_nodes=29)

        h = {'categorical': one_hot, 'integer': charges}
        
        # print(f"x in training: {x}")
        # print(f"h in training: {h}")
        # print(f"node_mask in training: {node_mask}")
        # print(f"edge_mask in training: {edge_mask}")
        # print(f"context in training: {context}")
        # print(f"log_pN in training: {log_pN}")
        # optim.zero_grad()
        # grad_fn = jax.value_and_grad(calculate_loss,
        #                          has_aux=True)
        # (_, acc), grads = grad_fn(state.params, state.apply_fn, batch)
        # state = state.apply_gradients(grads=grads)

        training = True
        rng, step_rng = jax.random.split(rng)
        args.counter=epoch
        #start_batch= time.time()
        state, loss, nll, reg_term, mean_abs_z, ema_state = train_step(step_rng, state, x, h, node_mask, edge_mask, context, log_pN, args.ode_regularization, ema_state)
        #print(f"Batch took {time.time() - start_batch:.1f} seconds.")

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        if i % args.n_report_steps == 0:
            pass
            # print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
            #       f"Loss {loss:.2f}, NLL: {nll:.2f}, "
            #       f"RegTerm: {reg_term.item():.1f}, "
            #       f"GradNorm: {grad_norm:.1f}")
        nll_epoch.append(nll.item())
        #####TODO Comment for fast processing
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
            # print("\n\n\n\n save and sample conditional!!!!!")
            start = time.time()
            #rng split
            rng, rng_sample = jax.random.split(rng, 2)
            #rng split
            if len(args.conditioning) > 0:
                save_and_sample_conditional(rng_sample, args, state, prop_dist, dataset_info, model, epoch=epoch)
            save_and_sample_chain(rng_sample, model_ema, args, dataset_info, prop_dist, model_state=state, epoch=epoch, batch_id=str(i))

            print(f'Sampling took {time.time() - start:.2f} seconds')

            vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
            vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
        #####TODO COmment for fast processing
        wandb.log({"Batch NLL": nll.item()}, commit=True)
        if args.break_train_epoch:
            break
    
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)
    return params, state, loss, ema_state


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            # pass
            # assert_correctly_masked(variable, node_mask)
            max_value = jnp.abs((variable * (1 - node_mask))).max()
    return max_value



def convet_property_norms_to_jax(property_norms):
    return {key: {sub_key: jnp.asarray(value) for sub_key, value in sub_dict.items()}
            for key, sub_dict in property_norms.items()}

@jax.jit
def test_step(rng, model_state, log_pN, x, node_mask, one_hot, context, edge_mask, charges):

    training = False
    # include_charges=False

    x = remove_mean_with_mask(x, node_mask)
    # assert_mean_zero_with_mask(x, node_mask)
    # max_value = check_mask_correct([x, one_hot, charges], node_mask)
    # assert max_value < 1e-4, "The maximum absolute value must be less than 1e-4"

    h = {'categorical': one_hot, 'integer': charges}


    rng, step_rng = jax.random.split(rng)
 
    h_int=h["integer"]
    h_cat=h["categorical"]
    nll, _, _ = losses.compute_loss_and_nll_test(
        step_rng,
        model_state,
        model_state.params,
        log_pN, 
        x, h_int, h_cat,
        node_mask,
        edge_mask, context
    )
    return nll

def test(rng, args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, model_state, partition='Test'):
    nll_epoch = 0
    n_samples = 0

    break_loop = False

    property_norms = convet_property_norms_to_jax(property_norms)

    n_iterations = len(loader)

    for i, data in enumerate(loader):

        node_mask = jnp.expand_dims(data['atom_mask'], 2)
        N = node_mask.squeeze(2).sum(1).astype(jnp.float32)
        log_pN = nodes_dist.log_prob(N)


        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context_jax(args.conditioning, data, property_norms)
        else:
            context = None


        dtype = jnp.float32
        x = data['positions'].astype(dtype)
        batch_size = x.shape[0]
        edge_mask = data['edge_mask'].astype(dtype)
        one_hot = data['one_hot'].astype(dtype)
        charges = (data['charges'] if args.include_charges else jnp.zeros(0)).astype(dtype)

        # x,node_mask,one_hot,context,edge_mask=pad_inputs(x,node_mask,one_hot,context,edge_mask)
        x, node_mask, one_hot, charges, context, edge_mask = pad_inputs(x, node_mask, one_hot, charges, context, edge_mask, max_nodes=29)

        nll=test_step(rng, model_state,log_pN, x,node_mask,one_hot,context,edge_mask,charges)



        nll_epoch += nll.item() * batch_size
        n_samples += batch_size
        if i % args.n_report_steps == 0:
            # print(f"{partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
            #       f"NLL: {nll_epoch / n_samples:.2f}")
            pass
        if break_loop:
            break

    return nll_epoch / n_samples


def save_and_sample_chain(rng, model, args, dataset_info, prop_dist, model_state,
                          epoch=0, id_from=0, batch_id=''):
    rng, rng_sasc = jax.random.split(rng)
    one_hot, charges, x = sample_chain(rng_sasc, args=args, flow=model, model_state=model_state,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(rng, model, nodes_dist, args, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples / batch_size)):
        rng, rng_sdsz, rng_s2 = jax.random.split(rng, 3)
        nodesxsample = nodes_dist.sample(rng_sdsz, batch_size)
        one_hot, charges, x, node_mask = sample(rng_s2, args, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule')


def analyze_and_save(rng, args, epoch, model_sample, nodes_dist,dataset_info, device,prop_dist, n_samples=1000,batch_size=100):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    #assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples / batch_size)):
        rng, rng_s3, rng_s4 = jax.random.split(rng,3)
        nodesxsample = nodes_dist.sample(rng_s3, batch_size)
        print(f"nodesxexample in analyze_and_save: {nodesxsample}")
        # one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                # nodesxsample=nodesxsample)
        one_hot, charges, x, node_mask = sample(rng_s4, args, model_sample, dataset_info, prop_dist, model_sample,
                                                nodesxsample=nodesxsample)

        # molecules['one_hot'].append(one_hot.detach().cpu())
        # molecules['x'].append(x.detach().cpu())
        # molecules['node_mask'].append(node_mask.detach().cpu())
        one_hot_np = np.array(jax.device_get(one_hot))
        x_np = np.array(jax.device_get(x))
        node_mask_np = np.array(jax.device_get(node_mask))

        one_hot_torch = torch.from_numpy(one_hot_np)
        x_torch = torch.from_numpy(x_np)
        node_mask_torch = torch.from_numpy(node_mask_np)

        molecules['one_hot'].append(one_hot_torch)
        molecules['x'].append(x_torch)
        molecules['node_mask'].append(node_mask_torch)

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)
    print("In analyze_and_save, log stability!")
    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict


def save_and_sample_conditional(rng, args, state, prop_dist, dataset_info, model, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(rng, args, state, dataset_info, prop_dist, model)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x
