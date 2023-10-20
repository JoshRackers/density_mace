###########################################################################################
# Data parsing utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import ase.data
import ase.io
import numpy as np
import h5py
import torch

from mace.tools import AtomicNumberTable
from e3nn import io

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]
Stress = np.ndarray  # [6, ]
Virials = np.ndarray  # [3,3]
Charges = np.ndarray  # [..., 1]
Dipoles = np.ndarray  # [..., 3]
Quadrupoles = np.ndarray # [..., 3, 3]
Octupoles = np.ndarray # [..., 3, 3, 3]
Cell = np.ndarray  # [3,3]
Pbc = tuple  # (3,)

DEFAULT_CONFIG_TYPE = "Default"
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom
    stress: Optional[Stress] = None  # eV/Angstrom^3
    virials: Optional[Virials] = None  # eV
    dipole: Optional[Vector] = None  # Debye
    charges: Optional[Charges] = None  # atomic unit
    dipoles: Optional[Dipoles] = None  # atomic unit
    quadrupoles: Optional[Quadrupoles] = None  # atomic unit
    octupoles: Optional[Octupoles] = None  # atomic unit
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None

    weight: float = 1.0  # weight of config in loss
    energy_weight: float = 1.0  # weight of config energy in loss
    forces_weight: float = 1.0  # weight of config forces in loss
    stress_weight: float = 1.0  # weight of config stress in loss
    virials_weight: float = 1.0  # weight of config virial in loss
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE  # config_type of config
    elements: Optional[str] = None


Configurations = List[Configuration]


def random_train_valid_split(
    items: Sequence, valid_fraction: float, seed: int
) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0

    size = len(items)
    train_size = size - int(valid_fraction * size)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )


def config_from_atoms_list(
    atoms_list: List[ase.Atoms],
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    config_type_weights: Dict[str, float] = None,
) -> Configurations:
    """Convert list of ase.Atoms into Configurations"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    all_configs = []
    for atoms in atoms_list:
        all_configs.append(
            config_from_atoms(
                atoms,
                energy_key=energy_key,
                forces_key=forces_key,
                stress_key=stress_key,
                virials_key=virials_key,
                dipole_key=dipole_key,
                charges_key=charges_key,
                config_type_weights=config_type_weights,
            )
        )
    return all_configs


def config_from_atoms(
    atoms: ase.Atoms,
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    config_type_weights: Dict[str, float] = None,
) -> Configuration:
    """Convert ase.Atoms to Configuration"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    energy = atoms.info.get(energy_key, None)  # eV
    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    stress = atoms.info.get(stress_key, None)  # eV / Ang
    virials = atoms.info.get(virials_key, None)
    dipole = atoms.info.get(dipole_key, None)  # Debye
    # Charges default to 0 instead of None if not found
    charges = atoms.arrays.get(charges_key, np.zeros(len(atoms)))  # atomic unit
    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
    )
    pbc = tuple(atoms.get_pbc())
    cell = np.array(atoms.get_cell())
    config_type = atoms.info.get("config_type", "Default")
    weight = atoms.info.get("config_weight", 1.0) * config_type_weights.get(
        config_type, 1.0
    )
    energy_weight = atoms.info.get("config_energy_weight", 1.0)
    forces_weight = atoms.info.get("config_forces_weight", 1.0)
    stress_weight = atoms.info.get("config_stress_weight", 1.0)
    virials_weight = atoms.info.get("config_virials_weight", 1.0)

    # fill in missing quantities but set their weight to 0.0
    if energy is None:
        energy = 0.0
        energy_weight = 0.0
    if forces is None:
        forces = np.zeros(np.shape(atoms.positions))
        forces_weight = 0.0
    if stress is None:
        stress = np.zeros(6)
        stress_weight = 0.0
    if virials is None:
        virials = np.zeros((3, 3))
        virials_weight = 0.0

    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        energy=energy,
        forces=forces,
        stress=stress,
        virials=virials,
        dipole=dipole,
        charges=charges,
        weight=weight,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
    )


def test_config_types(
    test_configs: Configurations,
) -> List[Tuple[Optional[str], List[Configuration]]]:
    """Split test set based on config_type-s"""
    test_by_ct = []
    all_cts = []
    for conf in test_configs:
        if conf.config_type not in all_cts:
            all_cts.append(conf.config_type)
            test_by_ct.append((conf.config_type, [conf]))
        else:
            ind = all_cts.index(conf.config_type)
            test_by_ct[ind][1].append(conf)
    return test_by_ct


def load_from_xyz(
    file_path: str,
    config_type_weights: Dict,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
    extract_atomic_energies: bool = False,
) -> Tuple[Dict[int, float], Configurations]:

    atoms_list = ase.io.read(file_path, index=":")

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    atomic_energies_dict = {}
    if extract_atomic_energies:
        atoms_without_iso_atoms = []

        for idx, atoms in enumerate(atoms_list):
            if len(atoms) == 1 and atoms.info["config_type"] == "IsolatedAtom":
                if energy_key in atoms.info.keys():
                    atomic_energies_dict[atoms.get_atomic_numbers()[0]] = atoms.info[
                        energy_key
                    ]
                else:
                    logging.warning(
                        f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                        "but does not contain an energy."
                    )
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info("Using isolated atom energies from training file")

        atoms_list = atoms_without_iso_atoms

    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
    )
    return atomic_energies_dict, configs


def load_from_hdf5(
    file_path: str,
    subset_key: str,
    config_type_weights: Dict,
    energy_key: str = "dft_total_energy",
    forces_key: str = "dft_total_gradient",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "mbis_charges",
    dipoles_key: str = "mbis_dipoles",
    quadrupoles_key: str = "mbis_quadrupoles",
    octupoles_key: str = "mbis_octupoles",
    extract_atomic_energies: bool = False,
) -> Tuple[Dict[int, float], Configurations]:

    logging.info("Extracting atomic energies is currently not supported for SPICE")
    atomic_energies_dict = {}
    # FUTURE: insert atomic energies stuff

    logging.warning("NOT USING MACE UNITS")
    logging.info("SPICE and multipoles datasets are all gas phase, not periodic")
    pbc = tuple([False,False,False])
    cell = np.array([[100,0,0],[0,100,0],[0,0,100]])
    config_type = "Default"
    weight = 1.0
    energy_weight  = 1.0
    forces_weight  = 1.0
    stress_weight  = 1.0
    virials_weight = 1.0

    h5_file = h5py.File(file_path)

    quad_carttensor = io.CartesianTensor("ij=ji")
    oct_carttensor = io.CartesianTensor("ijk=ikj=jki=jik=kij=kji")
    configs = []

    # for SPICE data
    if "SPICE" in file_path:
        for name in h5_file:
            h5data = h5_file[name]
            #print(h5data)
            if (subset_key == None) or (np.array(h5data["subset"]).item() == subset_key.encode('ascii')):
                #if len(np.array(h5data["atomic_numbers"])) == 4:
                    #for idx in range(1):
                    for idx in range(len(h5data["conformations"])):
                        #print(np.array(h5data["smiles"]))
                        #print(np.array(h5data[charges_key][idx,:,:]))
                        try:
                            configs.append(Configuration(
                                # replace these with general keys?
                                atomic_numbers=np.array(h5data["atomic_numbers"]),
                                positions=np.array(h5data["conformations"][idx,:,:]),
                                energy=np.array(h5data[energy_key][idx]),
                                forces=np.array(h5data[forces_key][idx,:,:]),
                                #stress=stress,
                                #virials=virials,
                                #dipole=dipole,
                                charges=np.array(h5data[charges_key][idx,:,:]),
                                dipoles=np.array(h5data[dipoles_key][idx,:,:]),
                                # assumes we've removed the trace
                                quadrupoles=np.array(h5data[quadrupoles_key][idx,:,:,:]),
                                # assumes we've removed the trace
                                octupoles=np.array(h5data[octupoles_key][idx,:,:,:,:]),

                                weight=weight,
                                energy_weight=energy_weight,
                                forces_weight=forces_weight,
                                stress_weight=stress_weight,
                                virials_weight=virials_weight,
                                config_type=config_type,
                                pbc=pbc,
                                cell=cell,
                            ))
                        except:
                            print("could not add this configuration. likely missing mbis multipoles")

        logging.info("Extracting atomic energies is currently not supported for SPICE")
        atomic_energies_dict = {}
        # FUTURE: insert atomic energies stuff
    else:
        print(">>> using multipoles dataset")
        proceed = True
        for i, key in enumerate(h5_file): # Iterates over each Unique Identifier
            # convert elements to atomic numbers
            short_periodic_table_dict = {b"H":1, b"C":6, b"N":7, b"O":8, b"F":9, b"S":16, b"CL":17, b"Cl":17}
            elements = h5_file[key]["elements"][()]
            atomic_numbers = [short_periodic_table_dict.get(element) for element in elements]
            #print(elements)
            #print(atomic_numbers)

            charges = np.array(h5_file[key]["monopoles"][()])
            dipoles = np.array(h5_file[key]["dipoles"][()])
            quadrupoles = np.array(h5_file[key]["traceless_quadrupoles"][()])
            octupoles=np.array(h5_file[key]["traceless_octupoles"][()])
            if charges.shape != (len(atomic_numbers), 1):
                proceed = False
                print("mono",charges.shape,len(atomic_numbers))
            elif dipoles.shape != (len(atomic_numbers), 3):
                proceed = False
                print("dip")
            elif quadrupoles.shape != (len(atomic_numbers), 3, 3):
                proceed = False
                print("quad")
            elif octupoles.shape != (len(atomic_numbers), 3, 3, 3):
                proceed = False
                print("oct")
            
            try:
                energy = np.array(h5_file[key]["relative_energy"][()])
            except:
                energy = np.array(h5_file[key]["energy"][()])

            

            if proceed == True:
                configs.append(Configuration(
                    elements=elements,
                    atomic_numbers=np.array(atomic_numbers),
                    positions=np.array(h5_file[key]["coordinates"][()]),
                    #energy=np.array(h5_file[key]["relative_energy"][()]),
                    energy=energy,
                    forces=-np.array(h5_file[key]["gradient"][()]),
                    charges=np.array(h5_file[key]["monopoles"][()]),
                    dipoles=np.array(h5_file[key]["dipoles"][()]),
                    quadrupoles=np.array(h5_file[key]["traceless_quadrupoles"][()]),
                    octupoles=np.array(h5_file[key]["traceless_octupoles"][()]),

                    weight=weight,
                    energy_weight=energy_weight,
                    forces_weight=forces_weight,
                    stress_weight=stress_weight,
                    virials_weight=virials_weight,
                    config_type=config_type,
                    pbc=pbc,
                    cell=cell,
                ))
            else:
                #print("missing multipoles for configuration",i,h5_file[key]["smiles"][()])
                print("missing multipoles for configuration",i)

            proceed = True

            # charges = np.array(h5_file[key]["monopoles"][()])
            # if charges.shape != (len(atomic_numbers), 1):
            #     print("charges shape is not correct!",i,charges.shape,len(atomic_numbers),atomic_numbers)
            #     print(charges)

            #assert charges is None or charges.shape == (len(atomic_numbers), 1) or charges.shape == (len(atomic_numbers),)


            # print("charges",np.array(h5_file[key]["monopoles"][()]).shape)
            # print("charges",np.array(h5_file[key]["monopoles"][()]))

            # if i > 100:
            #     break

    return atomic_energies_dict, configs


def compute_average_E0s(
    collections_train: Configurations, z_table: AtomicNumberTable
) -> Dict[int, float]:
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
    """
    len_train = len(collections_train)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    for i in range(len_train):
        B[i] = collections_train[i].energy
        for j, z in enumerate(z_table.zs):
            A[i, j] = np.count_nonzero(collections_train[i].atomic_numbers == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.warning(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = 0.0
    return atomic_energies_dict
