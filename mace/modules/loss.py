###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import torch

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch

from e3nn import io


def mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    return torch.mean(torch.square(ref["energy"] - pred["energy"]))  # []


def weighted_mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight  # [n_graphs, ]
    configs_energy_weight = ref.energy_weight  # [n_graphs, ]
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # [n_graphs,]
    return torch.mean(
        configs_weight
        * configs_energy_weight
        * torch.square((ref["energy"] - pred["energy"]) / num_atoms)
    )  # []


def weighted_mean_squared_stress(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight.view(-1, 1, 1)  # [n_graphs, ]
    configs_stress_weight = ref.stress_weight.view(-1, 1, 1)  # [n_graphs, ]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)  # [n_graphs,]
    return torch.mean(
        configs_weight
        * configs_stress_weight
        * torch.square((ref["stress"] - pred["stress"]) / num_atoms)
    )  # []


def weighted_mean_squared_virials(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight.view(-1, 1, 1)  # [n_graphs, ]
    configs_virials_weight = ref.virials_weight.view(-1, 1, 1)  # [n_graphs, ]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)  # [n_graphs,]
    return torch.mean(
        configs_weight
        * configs_virials_weight
        * torch.square((ref["virials"] - pred["virials"]) / num_atoms)
    )  # []


def mean_squared_error_forces(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # forces: [n_atoms, 3]
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    return torch.mean(
        configs_weight
        * configs_forces_weight
        * torch.square(ref["forces"] - pred["forces"])
    )  # []


def weighted_mean_squared_error_dipole(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # dipole: [n_graphs, ]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)  # [n_graphs,1]
    return torch.mean(torch.square((ref["dipole"] - pred["dipole"]) / num_atoms))  # []
    # return torch.mean(torch.square((torch.reshape(ref['dipole'], pred["dipole"].shape) - pred['dipole']) / num_atoms))  # []


def mean_squared_error_multipole(ref: Batch, pred: TensorDict, moment: str, device=None, cart = None, rtp = None) -> torch.Tensor:
    multiplier = 1
    ref_cart = ref[moment]
    if moment == "charges":
        pred_cart = pred[moment]
        #multiplier = 0
    else:
        if moment == "dipoles":
            #cart = io.CartesianTensor("i")
            #pred_cart = cart.to_cartesian(pred[moment])
            pred_cart = pred[moment]
            #multiplier = 0
        elif moment == "quadrupoles":
            #cart = io.CartesianTensor("ij=ji")
            # this definition still has a trace, which to_cartesian will expect
            # so we have to add a zero to the front
            zeros = torch.zeros(len(pred[moment]), device=device)
            pred_trace = torch.cat((zeros.unsqueeze(-1),pred[moment]),dim=-1)
            pred_cart = cart.to_cartesian(pred_trace, rtp=rtp)
            # print("qudrupole loss")
            # print("ref")
            # print(ref_cart)
            # print("pred")
            # print(pred_cart)
        elif moment == "octupoles":
            #  see decomposition of rank-3 tensor here:
            # https://math.stackexchange.com/questions/3585336/finding-the-irreducible-components-of-a-rank-3-tensor
            #cart = io.CartesianTensor("ijk=ikj=jki=jik=kij=kji")
            zeros = torch.zeros([len(pred[moment]),3], device=device)
            pred_trace = torch.cat((zeros,pred[moment]),dim=-1)
            pred_cart = cart.to_cartesian(pred_trace, rtp=rtp)
            # print("octupole loss")
            # print("ref")
            # print(ref_cart)
            # print("pred")
            # print(pred_cart)
    
    return torch.mean(torch.square(ref_cart - pred_cart)) * multiplier


class EnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return self.energy_weight * mean_squared_error_energy(
            ref, pred
        ) + self.forces_weight * mean_squared_error_forces(ref, pred)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return self.energy_weight * weighted_mean_squared_error_energy(
            ref, pred
        ) + self.forces_weight * mean_squared_error_forces(ref, pred)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


class WeightedForcesLoss(torch.nn.Module):
    def __init__(self, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return self.forces_weight * mean_squared_error_forces(ref, pred)

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"forces_weight={self.forces_weight:.3f})"


class WeightedEnergyForcesStressLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.stress_weight * weighted_mean_squared_stress(ref, pred)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedEnergyForcesVirialsLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, virials_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "virials_weight",
            torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.virials_weight * weighted_mean_squared_virials(ref, pred)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, virials_weight={self.virials_weight:.3f})"
        )


class DipoleSingleLoss(torch.nn.Module):
    def __init__(self, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.dipole_weight * weighted_mean_squared_error_dipole(ref, pred) * 100.0
        )  # multiply by 100 to have the right scale for the loss

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"dipole_weight={self.dipole_weight:.3f})"


class WeightedEnergyForcesDipoleLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.dipole_weight * weighted_mean_squared_error_dipole(ref, pred) * 100
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipole_weight:.3f})"
        )


class MultipolesLoss(torch.nn.Module):
    def __init__(self, device, highest_multipole_moment: int, multipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "multipole_weight",
            torch.tensor(multipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "highest_multipole_moment",
            torch.tensor(highest_multipole_moment, dtype=torch.get_default_dtype()),
        )
        self.device = device
        quad_cart = io.CartesianTensor("ij=ji")
        self.quadrupole_cartesiantensor = quad_cart
        self.quadrupole_rtp = quad_cart.reduced_tensor_products().to(device, dtype=torch.get_default_dtype())
        oct_cart = io.CartesianTensor("ijk=ikj=jki=jik=kij=kji")
        self.octupole_cartesiantensor = oct_cart
        self.octupole_rtp = oct_cart.reduced_tensor_products().to(device, dtype=torch.get_default_dtype())


    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        if self.highest_multipole_moment >= 0:
            error = self.multipole_weight * mean_squared_error_multipole(ref, pred, device=self.device, moment="charges")
        if self.highest_multipole_moment >= 1:
            error += self.multipole_weight * mean_squared_error_multipole(ref, pred, device=self.device, moment="dipoles")
        if self.highest_multipole_moment >= 2:
            error += self.multipole_weight * mean_squared_error_multipole(ref, pred, device=self.device, moment="quadrupoles", cart = self.quadrupole_cartesiantensor, rtp = self.quadrupole_rtp)
        if self.highest_multipole_moment >= 3:
            error += self.multipole_weight * mean_squared_error_multipole(ref, pred, device=self.device, moment="octupoles", cart = self.octupole_cartesiantensor, rtp = self.octupole_rtp)

        return (error)

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"multipole_weight={self.multipole_weight:.3f})"
    

class WeightedEnergyForcesMultipolesLoss(torch.nn.Module):
    def __init__(self, device, highest_multipole_moment: int, energy_weight=1.0, forces_weight=1.0, multipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "highest_multipole_moment",
            torch.tensor(highest_multipole_moment, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "multipole_weight",
            torch.tensor(multipole_weight, dtype=torch.get_default_dtype()),
        )
        self.device = device
        quad_cart = io.CartesianTensor("ij=ji")
        self.quadrupole_cartesiantensor = quad_cart
        self.quadrupole_rtp = quad_cart.reduced_tensor_products().to(device, dtype=torch.get_default_dtype())
        oct_cart = io.CartesianTensor("ijk=ikj=jki=jik=kij=kji")
        self.octupole_cartesiantensor = oct_cart
        self.octupole_rtp = oct_cart.reduced_tensor_products().to(device, dtype=torch.get_default_dtype())

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        energy_error = self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
        forces_error = self.forces_weight * mean_squared_error_forces(ref, pred)

        if self.highest_multipole_moment >= 0:
            multipole_error = self.multipole_weight * mean_squared_error_multipole(ref, pred, device=self.device, moment="charges")
        if self.highest_multipole_moment >= 1:
            multipole_error += self.multipole_weight * mean_squared_error_multipole(ref, pred, device=self.device, moment="dipoles")
        if self.highest_multipole_moment >= 2:
            multipole_error += self.multipole_weight * mean_squared_error_multipole(ref, pred, device=self.device, moment="quadrupoles", cart = self.quadrupole_cartesiantensor, rtp = self.quadrupole_rtp)
        if self.highest_multipole_moment >= 3:
            multipole_error += self.multipole_weight * mean_squared_error_multipole(ref, pred, device=self.device, moment="octupoles", cart = self.octupole_cartesiantensor, rtp = self.octupole_rtp)

        # print("energy:",energy_error)
        # print("forces:",forces_error)
        # print("multipoles:",multipole_error)

        return (energy_error + forces_error + multipole_error)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, multipole_weight={self.multipole_weight:.3f})"
        )