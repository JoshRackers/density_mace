###########################################################################################
# Training script
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from e3nn import io

import wandb

from . import torch_geometric
from .checkpoint import CheckpointHandler, CheckpointState
from .torch_tools import tensor_dict_to_device, to_numpy
from .utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
)


@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int
    loss_fn: torch.nn.Module


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    use_wandb: bool,
    eval_interval: int,
    output_args: Dict[str, bool],
    device: torch.device,
    log_errors: str,
    swa: Optional[SWAContainer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
    highest_multipole_moment: int = None,
):
    
    if use_wandb:
        wandb.watch(model)

    lowest_loss = np.inf
    patience_counter = 0
    swa_start = True

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")
    logging.info("Started training")
    for epoch in range(start_epoch, max_num_epochs):
        # Train
        cumulative_loss = 0.0
        for batch in train_loader:
            step_loss, opt_metrics = take_step(
                model=model,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                ema=ema,
                output_args=output_args,
                max_grad_norm=max_grad_norm,
                device=device,
            )
            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch
            logger.log(opt_metrics)

            cumulative_loss += step_loss.detach()

        average_loss = cumulative_loss/len(train_loader) 
        if use_wandb:
            wandb.log({"epoch": epoch, "Train Loss": average_loss})

        # Validate
        if epoch % eval_interval == 0:
            if ema is not None:
                with ema.average_parameters():
                    valid_loss, eval_metrics = evaluate(
                        model=model,
                        loss_fn=loss_fn,
                        data_loader=valid_loader,
                        output_args=output_args,
                        device=device,
                    )
            else:
                valid_loss, eval_metrics = evaluate(
                    model=model,
                    loss_fn=loss_fn,
                    data_loader=valid_loader,
                    output_args=output_args,
                    device=device,
                )
            eval_metrics["mode"] = "eval"
            eval_metrics["epoch"] = epoch
            logger.log(eval_metrics)
            if log_errors == "PerAtomRMSE":
                error_e = eval_metrics["rmse_e_per_atom"] * 1e3
                error_f = eval_metrics["rmse_f"] * 1e3
                logging.info(
                    f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
                )
            elif (
                log_errors == "PerAtomRMSEstressvirials"
                and eval_metrics["rmse_stress_per_atom"] is not None
            ):
                error_e = eval_metrics["rmse_e_per_atom"] * 1e3
                error_f = eval_metrics["rmse_f"] * 1e3
                error_stress = eval_metrics["rmse_stress_per_atom"] * 1e3
                logging.info(
                    f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_stress_per_atom={error_stress:.1f} meV / A"
                )
            elif (
                log_errors == "PerAtomRMSEstressvirials"
                and eval_metrics["rmse_virials_per_atom"] is not None
            ):
                error_e = eval_metrics["rmse_e_per_atom"] * 1e3
                error_f = eval_metrics["rmse_f"] * 1e3
                error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
                logging.info(
                    f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_virials_per_atom={error_virials:.1f} meV / A"
                )
            elif log_errors == "TotalRMSE":
                error_e = eval_metrics["rmse_e"] * 1e3
                error_f = eval_metrics["rmse_f"] * 1e3
                logging.info(
                    f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
                )
            elif log_errors == "PerAtomMAE":
                error_e = eval_metrics["mae_e_per_atom"] * 1e3
                error_f = eval_metrics["mae_f"] * 1e3
                logging.info(
                    f"Epoch {epoch}: loss={valid_loss:.4f}, MAE_E_per_atom={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A"
                )
            elif log_errors == "TotalMAE":
                error_e = eval_metrics["mae_e"] * 1e3
                error_f = eval_metrics["mae_f"] * 1e3
                logging.info(
                    f"Epoch {epoch}: loss={valid_loss:.4f}, MAE_E={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A"
                )
            elif log_errors == "DipoleRMSE":
                error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
                logging.info(
                    f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.2f} mDebye"
                )
            elif log_errors == "MultipolesRMSE":
                if highest_multipole_moment >= 0:
                    error_charges = eval_metrics["rmse_charges_per_atom"]
                    logging.info(
                        f"Epoch {epoch}: validloss={valid_loss:.4f}, RMSE_charges_per_atom={error_charges:.2f} "
                    )
                if highest_multipole_moment >= 1:
                    error_dipoles = eval_metrics["rmse_dipoles_per_atom"]
                    logging.info(
                        f"Epoch {epoch}: validloss={valid_loss:.4f}, RMSE_dipoles_per_atom={error_dipoles:.2f} "
                    )
                if highest_multipole_moment >= 2:
                    error_quadrupoles = eval_metrics["rmse_quadrupoles_per_atom"]
                    logging.info(
                        f"Epoch {epoch}: validloss={valid_loss:.4f}, RMSE_quadrupoles_per_atom={error_quadrupoles:.2f} "
                    )
                if highest_multipole_moment >= 3:
                    error_octupoles = eval_metrics["rmse_octupoles_per_atom"]
                    logging.info(
                        f"Epoch {epoch}: validloss={valid_loss:.4f}, RMSE_octupoles_per_atom={error_octupoles:.2f} "
                    )
            elif log_errors == "EnergyDipoleRMSE":
                error_e = eval_metrics["rmse_e_per_atom"] * 1e3
                error_f = eval_metrics["rmse_f"] * 1e3
                error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
                logging.info(
                    f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_Mu_per_atom={error_mu:.2f} mDebye"
                )
            if valid_loss >= lowest_loss:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(
                        f"Stopping optimization after {patience_counter} epochs without improvement"
                    )
                    break
            else:
                lowest_loss = valid_loss
                patience_counter = 0
                if ema is not None:
                    with ema.average_parameters():
                        checkpoint_handler.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                        )
                else:
                    checkpoint_handler.save(
                        state=CheckpointState(model, optimizer, lr_scheduler),
                        epochs=epoch,
                    )
            
            if use_wandb:
                wandb.log(eval_metrics)

        # LR scheduler and SWA update
        if swa is None or epoch < swa.start:
            lr_scheduler.step(valid_loss)  # Can break if exponential LR, TODO fix that!
        else:
            if swa_start:
                logging.info("Changing loss based on SWA")
                swa_start = False
            loss_fn = swa.loss_fn
            swa.model.update_parameters(model)
            swa.scheduler.step()

    logging.info("Training complete")


def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:

    start_time = time.time()
    batch = batch.to(device)
    optimizer.zero_grad()
    output = model(
        batch,
        training=True,
        compute_force=output_args["forces"],
        compute_virials=output_args["virials"],
        compute_stress=output_args["stress"],
    )

    #print("train, output",output["charges"])
    #print("train, batch ", batch["charges"])
    loss = loss_fn(pred=output, ref=batch)
    
    torch.set_printoptions(precision=10)
    try:
        print("energy, ref:", batch["energy"]," pred:", output["energy"])
    except:
        pass

    try:
        print(" ref charge",batch["charges"])
        print("pred charge",output["charges"].data)
    except:
        pass

    try:
        print(" ref dipoles",batch["dipoles"])
        print("pred dipoles",output["dipoles"].data)
    except:
        pass

    print("loss",loss.data)

    # try:
    #     print(" ref quadrupoles",batch["quadrupoles"])
    #     print("pred quadrupoles",output["quadrupoles"].data)
    # except:
    #     pass

    

    #print(" ref charge",batch["charges"][0,:],"dipole",batch["dipoles"][0,:])
    #print("pred charge",output["charges"][0,:].data,"dipole",output["dipoles"][0,:].data)
    #print(output["charges"].shape,batch["charges"].shape)
    #print(output["dipoles"].shape,batch["dipoles"].shape)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


def evaluate(
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        data_loader: DataLoader,
        output_args: Dict[str, bool],
        device: torch.device,
    ) -> Tuple[float, Dict[str, Any]]:

    total_loss = 0.0
    E_computed = False
    delta_es_list = []
    delta_es_per_atom_list = []
    delta_fs_list = []
    Fs_computed = False
    fs_list = []
    stress_computed = False
    delta_stress_list = []
    delta_stress_per_atom_list = []
    virials_computed = False
    delta_virials_list = []
    delta_virials_per_atom_list = []
    Mus_computed = False
    delta_mus_list = []
    delta_mus_per_atom_list = []
    mus_list = []
    charges_computed = False
    delta_charges_per_atom_list = []
    charges_list = []
    dipoles_computed = False
    delta_dipoles_per_atom_list = []
    dipoles_list = []
    quadrupoles_computed = False
    delta_quadrupoles_per_atom_list = []
    quadrupoles_list = []
    octupoles_computed = False
    delta_octupoles_per_atom_list = []
    octupoles_list = []
    batch = None  # for pylint

    start_time = time.time()

    quadrupole_cartesiantensor = io.CartesianTensor("ij=ji")
    quadrupole_rtp = quadrupole_cartesiantensor.reduced_tensor_products().to(device, dtype=torch.get_default_dtype())
    octupole_cartesiantensor = io.CartesianTensor("ijk=ikj=jki=jik=kij=kji")
    octupole_rtp = octupole_cartesiantensor.reduced_tensor_products().to(device, dtype=torch.get_default_dtype())


    for batch in data_loader:
        batch = batch.to(device)
        output = model(
            batch,
            training=False,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        batch = batch.cpu()
        output = tensor_dict_to_device(output, device=torch.device("cpu"))

        loss = loss_fn(pred=output, ref=batch)
        total_loss += to_numpy(loss).item()

        if output.get("energy") is not None and batch.energy is not None:
            E_computed = True
            delta_es_list.append(batch.energy - output["energy"])
            delta_es_per_atom_list.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
        if output.get("forces") is not None and batch.forces is not None:
            Fs_computed = True
            delta_fs_list.append(batch.forces - output["forces"])
            fs_list.append(batch.forces)
        if output.get("stress") is not None and batch.stress is not None:
            stress_computed = True
            delta_stress_list.append(batch.stress - output["stress"])
            delta_stress_per_atom_list.append(
                (batch.stress - output["stress"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("virials") is not None and batch.virials is not None:
            virials_computed = True
            delta_virials_list.append(batch.virials - output["virials"])
            delta_virials_per_atom_list.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("dipole") is not None and batch.dipole is not None:
            Mus_computed = True
            delta_mus_list.append(batch.dipole - output["dipole"])
            delta_mus_per_atom_list.append(
                (batch.dipole - output["dipole"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )
            mus_list.append(batch.dipole)
        if output.get("charges") is not None and batch.charges is not None:
            charges_computed = True
            delta_charges_per_atom_list.append(batch.charges - output["charges"])
            charges_list.append(batch.charges)
        if output.get("dipoles") is not None and batch.dipoles is not None:
            dipoles_computed = True
            delta_dipoles_per_atom_list.append(batch.dipoles - output["dipoles"])
            dipoles_list.append(batch.dipoles)
        if output.get("quadrupoles") is not None and batch.quadrupoles is not None:
            quadrupoles_computed = True
            # switch to cartesian
            #cart = io.CartesianTensor("ij=ji")
            zeros = torch.zeros(len(output["quadrupoles"]))
            add_trace = torch.cat((zeros.unsqueeze(-1),output["quadrupoles"]),dim=-1)
            #quad_cart = cart.to_cartesian(add_trace)
            quad_cart = quadrupole_cartesiantensor.to_cartesian(add_trace, rtp=quadrupole_rtp)
            delta_quadrupoles_per_atom_list.append(batch.quadrupoles - quad_cart)
            quadrupoles_list.append(batch.quadrupoles)
        if output.get("octupoles") is not None and batch.octupoles is not None:
            octupoles_computed = True
            #cart = io.CartesianTensor("ijk=ikj=jki=jik=kij=kji")
            zeros = torch.zeros([len(output["octupoles"]),3])
            add_trace = torch.cat((zeros,output["octupoles"]),dim=-1)
            #oct_cart = cart.to_cartesian(add_trace)
            oct_cart = octupole_cartesiantensor.to_cartesian(add_trace, rtp=octupole_rtp)

            delta_octupoles_per_atom_list.append(batch.octupoles - oct_cart)
            octupoles_list.append(batch.octupoles)
        

    avg_loss = total_loss / len(data_loader)

    aux = {
        "loss": avg_loss,
    }

    if E_computed:
        delta_es = to_numpy(torch.cat(delta_es_list, dim=0))
        delta_es_per_atom = to_numpy(torch.cat(delta_es_per_atom_list, dim=0))
        aux["mae_e"] = compute_mae(delta_es)
        aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
        aux["rmse_e"] = compute_rmse(delta_es)
        aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
        aux["q95_e"] = compute_q95(delta_es)
    if Fs_computed:
        delta_fs = to_numpy(torch.cat(delta_fs_list, dim=0))
        fs = to_numpy(torch.cat(fs_list, dim=0))
        aux["mae_f"] = compute_mae(delta_fs)
        aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
        aux["rmse_f"] = compute_rmse(delta_fs)
        aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
        aux["q95_f"] = compute_q95(delta_fs)
    if stress_computed:
        delta_stress = to_numpy(torch.cat(delta_stress_list, dim=0))
        delta_stress_per_atom = to_numpy(torch.cat(delta_stress_per_atom_list, dim=0))
        aux["mae_stress"] = compute_mae(delta_stress)
        aux["rmse_stress"] = compute_rmse(delta_stress)
        aux["rmse_stress_per_atom"] = compute_rmse(delta_stress_per_atom)
        aux["q95_stress"] = compute_q95(delta_stress)
    if virials_computed:
        delta_virials = to_numpy(torch.cat(delta_virials_list, dim=0))
        delta_virials_per_atom = to_numpy(torch.cat(delta_virials_per_atom_list, dim=0))
        aux["mae_virials"] = compute_mae(delta_virials)
        aux["rmse_virials"] = compute_rmse(delta_virials)
        aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
        aux["q95_virials"] = compute_q95(delta_virials)
    if Mus_computed:
        delta_mus = to_numpy(torch.cat(delta_mus_list, dim=0))
        delta_mus_per_atom = to_numpy(torch.cat(delta_mus_per_atom_list, dim=0))
        mus = to_numpy(torch.cat(mus_list, dim=0))
        aux["mae_mu"] = compute_mae(delta_mus)
        aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
        aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
        aux["rmse_mu"] = compute_rmse(delta_mus)
        aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
        aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
        aux["q95_mu"] = compute_q95(delta_mus)
    if charges_computed:
        delta_charges_per_atom = to_numpy(torch.cat(delta_charges_per_atom_list, dim=0))
        charges = to_numpy(torch.cat(charges_list, dim=0))
        aux["mae_charges_per_atom"] = compute_mae(delta_charges_per_atom)
        aux["rmse_charges_per_atom"] = compute_rmse(delta_charges_per_atom)
    if dipoles_computed:
        delta_dipoles_per_atom = to_numpy(torch.cat(delta_dipoles_per_atom_list, dim=0))
        dipoles = to_numpy(torch.cat(dipoles_list, dim=0))
        aux["mae_dipoles_per_atom"] = compute_mae(delta_dipoles_per_atom)
        aux["rmse_dipoles_per_atom"] = compute_rmse(delta_dipoles_per_atom)
    if quadrupoles_computed:
        delta_quadrupoles_per_atom = to_numpy(torch.cat(delta_quadrupoles_per_atom_list, dim=0))
        quadrupoles = to_numpy(torch.cat(quadrupoles_list, dim=0))
        aux["mae_quadrupoles_per_atom"] = compute_mae(delta_quadrupoles_per_atom)
        aux["rmse_quadrupoles_per_atom"] = compute_rmse(delta_quadrupoles_per_atom)
    if octupoles_computed:
        delta_octupoles_per_atom = to_numpy(torch.cat(delta_octupoles_per_atom_list, dim=0))
        octupoles = to_numpy(torch.cat(octupoles_list, dim=0))
        aux["mae_octupoles_per_atom"] = compute_mae(delta_octupoles_per_atom)
        aux["rmse_octupoles_per_atom"] = compute_rmse(delta_octupoles_per_atom)

    aux["time"] = time.time() - start_time

    return avg_loss, aux
