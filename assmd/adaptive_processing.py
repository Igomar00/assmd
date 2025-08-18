#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 10:44:02 2025

@author: igor
"""

import os
import shutil
import logging
import glob
import subprocess
import pickle
import numpy as np
import deeptime as dt
import pytraj as pt
from typing import List, Dict, Tuple
import multiprocessing as mp

from assmd import file_structure as fs
from assmd import config as conf


def check_submitit_output(
    submitit_results: List[Tuple[int, str]],
) -> Tuple[bool, Dict[str, str]]:
    """
    Verify if submitit outputs indicate successful completion.

    Args:
        submitit_results: List of tuples (exit_code, command) from launch_md function

    Returns:
        Tuple of (success_status, error_details)
    """
    success = True
    error_details = {}

    for idx, (exit_code, command) in enumerate(submitit_results):
        if exit_code != 0:
            success = False
            error_details[f"simulation_{idx}"] = (
                f"Failed with exit code {exit_code}. Command: {command}"
            )

    return success, error_details


def find_simulation_files(
    workspace: fs.AdaptiveWorkplace, seed_num, epoch_num
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Find essential simulation files in a directory.

    Args:
        sim_dir: Path to simulation directory

    Returns:
        Tuple of (found_files, missing_files) where each is a dictionary:
        - found_files: maps file types to their paths
        - missing_files: maps file types to their expected paths
    """

    sim_path = workspace.get_files(
        workspace.get_files_by_tags(
            [f"run_dir_epoch_{epoch_num}", f"run_seed_{seed_num}"]
        )
    ).abs_path
    # Expected files with 'prod' prefix
    expected_files = {
        "log": os.path.join(sim_path, "prod_log.out"),
        "coords": os.path.join(sim_path, "prod_coords.rst"),
        "traj": os.path.join(sim_path, "prod_traj.nc"),
    }

    # Find topology file (allowing for pattern matching)
    topology_files = glob.glob(os.path.join(sim_path, "topology*"))
    if topology_files:
        expected_files["topology"] = topology_files[0]

    # Separate into found and missing
    found_files = {}
    missing_files = {}

    for file_type, file_path in expected_files.items():
        if os.path.exists(file_path):
            found_files[file_type] = str(file_path)
        else:
            missing_files[file_type] = str(file_path)

    return found_files, missing_files


def verify_trajectory(
    traj_path: str, top_path: str, expected_frames: int
) -> Tuple[bool, str]:
    """
    Verify if a trajectory file can be loaded and has correct number of frames.

    Args:
        traj_path: Path to trajectory file
        top_path: Path to topology file
        expected_frames: Expected number of frames

    Returns:
        Tuple of (success_status, error_message)
    """
    try:
        # Attempt to load trajectory
        traj = pt.load(traj_path, top=top_path)

        # Check number of frames
        n_frames = traj.n_frames
        if n_frames != expected_frames:
            return False, f"Expected {expected_frames} frames, found {n_frames}"

        # Basic sanity checks
        if traj.n_atoms == 0:
            return False, "Trajectory contains 0 atoms"

        return True, ""

    except Exception as e:
        return False, f"Failed to load trajectory: {str(e)}"


def validate_simulation_directories(
    workspace: fs.AdaptiveWorkplace,
    config: conf.JobConfig,
    epoch_num: int,
    expected_frames: int,
    submitit_results: List[Tuple[int, str]],
) -> Dict:
    """
    Validate multiple simulation directories.

    Args:
        sim_dirs: List of simulation directory paths
        expected_frames: Expected number of frames in trajectories
        submitit_results: List of (exit_code, command) tuples from launch_md

    Returns:
        Dictionary containing validation results
    """
    results = {"overall_success": True, "details": {"submitit": {}, "simulations": {}}}

    # First check submitit outputs
    submitit_success, submitit_errors = check_submitit_output(submitit_results)
    results["details"]["submitit"] = {
        "success": submitit_success,
        "errors": submitit_errors,
    }

    # Only proceed with file checks if submitit was successful
    if not submitit_success:
        results["overall_success"] = False
        return results

    # Check each simulation directory
    for seed_num in range(config.general.num_seeds):
        dir_results = {"success": True, "errors": []}

        # Find simulation files
        files = find_simulation_files(workspace, seed_num, epoch_num)

        # Check if we found all required files
        required_files = {"log", "coords", "traj", "topology"}
        if not all(k in files[0] for k in required_files):
            missing = files[1]
            dir_results["success"] = False
            dir_results["errors"].append(
                f"Missing required files: {', '.join(missing)}"
            )
            continue

        # register found files
        workspace.add_file(
            files[0]["log"],
            tags=[f"seed_{seed_num}", f"run_epoch_{epoch_num}", "prod_log"],
        )
        workspace.add_file(
            files[0]["traj"],
            tags=[f"seed_{seed_num}", f"run_epoch_{epoch_num}", "prod_traj"],
        )
        workspace.add_file(
            files[0]["coords"],
            tags=[f"seed_{seed_num}", f"run_epoch_{epoch_num}", "prod_restart"],
        )

        # Check trajectory
        traj_success, traj_error = verify_trajectory(
            files[0]["traj"], files[0]["topology"], expected_frames
        )
        if not traj_success:
            dir_results["success"] = False
            dir_results["errors"].append(f"Trajectory error: {traj_error}")

        results["details"]["simulations"][seed_num] = dir_results
        if not dir_results["success"]:
            results["overall_success"] = False

    if workspace.check_all_files():
        results["overall_success"] = False

    return results


def extract_high_probability_indices(
    memberships, threshold=0.5, ensure_assignment=False
):
    """
    Extract indices of microstates that belong to macrostates with probability >= threshold

    Parameters:
    -----------
    memberships : numpy.ndarray
        A matrix M of shape (n,m) where n is the number of microstates and m is the
        number of macrostates. M[i,j] represents the probability of microstate i
        belonging to macrostate j.
    threshold : float, optional
        The probability threshold (default: 0.5)
    ensure_assignment : bool, optional
        If True, ensure each macrostate has at least one microstate assigned to it,
        even if no microstate exceeds the threshold (default: False)

    Returns:
    --------
    dict
        A dictionary where keys are macrostate indices and values are lists of
        microstate indices that belong to that macrostate with probability >= threshold
        or highest available probability if ensure_assignment=True
    """
    n_microstates, n_macrostates = memberships.shape
    result = {}
    assigned = set()  # Track all assigned microstates

    # First pass: assign microstates that exceed the threshold
    for macrostate in range(n_macrostates):
        # Get indices of microstates where probability >= threshold for this macrostate
        indices = np.where(memberships[:, macrostate] >= threshold)[0].tolist()
        result[macrostate] = indices
        assigned.update(indices)

    # Second pass (if needed): ensure each macrostate has at least one microstate
    if ensure_assignment:
        for macrostate in range(n_macrostates):
            if len(result[macrostate]) == 0:
                # Create a copy of the column for this macrostate
                probs = memberships[:, macrostate].copy()

                # Set probabilities of already assigned microstates to -1 (to exclude them)
                for idx in assigned:
                    probs[idx] = -1

                # If all microstates are assigned, we might need to relax the constraint
                if np.all(probs < 0):
                    print(
                        f"Warning: All microstates already assigned. Macrostate {macrostate} "
                        "will use a microstate already assigned to another macrostate."
                    )
                    # Reset probabilities to original values
                    probs = memberships[:, macrostate].copy()

                # Find microstate with highest probability for this macrostate
                best_microstate = np.argmax(probs)

                # Assign this microstate
                result[macrostate] = [best_microstate]
                assigned.add(best_microstate)

    return result


def get_assigned_microstates(memberships, threshold=0.5):
    """
    Get all microstate indices that belong to any macrostate with probability >= threshold.
    If a macrostate has no microstates above the threshold, assign the microstate with the
    highest probability that is not already assigned to another macrostate.

    Parameters:
    -----------
    memberships : numpy.ndarray
        The memberships matrix
    threshold : float, optional
        The probability threshold (default: 0.5)

    Returns:
    --------
    tuple: (list, dict)
        - list: Sorted list of all assigned microstate indices
        - dict: Dictionary mapping macrostate indices to their assigned microstate indices
    """
    # Use the enhanced extract_high_probability_indices function with ensure_assignment=True
    macrostate_assignments = extract_high_probability_indices(
        memberships, threshold=threshold, ensure_assignment=True
    )

    # Get all unique assigned microstates
    assigned = set()
    for indices in macrostate_assignments.values():
        assigned.update(indices)

    return sorted(list(assigned)), macrostate_assignments


# functions from HTMD -----


def getNumMicrostates(numFrames):  # adapted from HTMD
    """Heuristic that calculates number of clusters from number of frames"""
    K = int(max(np.round(0.6 * np.log10(numFrames / 1000) * 1000 + 50), 100))
    if K > numFrames / 3:  # Ugly patch for low-data regimes ...
        K = int(numFrames / 3)
    return K


def getNumMacrostates(config: conf.JobConfig, data, num_micro):  # adapted from HTMD
    """Heuristic for calculating the number of macrostates for the Markov model"""
    macronum = config.ligand_model.num_macro
    if num_micro < macronum:
        macronum = int(np.ceil(num_micro / 2))
    # Calculating how many timescales are above the lag time to limit number of macrostates
    counts = dt.markov.TransitionCountEstimator(
        config.ligand_model.markov_lag, "sliding"
    ).fit_fetch(data)
    model = dt.markov.msm.MaximumLikelihoodMSM(
        allow_disconnected=False, use_lcc=False
    ).fit_fetch(counts.submodel_largest())
    its_data = dt.util.validation.implied_timescales(model, n_its=macronum)
    timesc = its_data._its
    macronum = min(
        np.sum(timesc > config.ligand_model.markov_lag), config.general.num_seeds
    )
    return macronum


def validateEpoch(workspace, config, epoch_num, prod_num_frames, submitit_results):
    logger = logging.getLogger(__name__)

    validation_results = validate_simulation_directories(
        workspace, config, epoch_num, prod_num_frames, submitit_results
    )
    if validation_results["overall_success"]:  # redo to act on paths instead of seednum
        logger.info(f"Result files for epoch {epoch_num} seem alright :D")
    else:
        if validation_results["details"]["submitit"]["success"]:
            logger.critical("Validation of simulation files failed")
            for sim_dir in validation_results["details"]["simulations"]:
                if not validation_results["details"]["simulations"][sim_dir]["success"]:
                    logger.critical(f"in simulation directory: {sim_dir}")
                    for error in validation_results["details"]["simulations"][sim_dir][
                        "errors"
                    ]:
                        logger.critical(error)
        else:
            logger.critical("Submitit failed for jobs:")
            for error in validation_results["details"]["submitit"]["errors"]:
                logger.critical(error)
        return False

    if workspace.check_all_files():
        logger.critical(f"File integrity fault detected after run of epoch {epoch_num}")
        return False

    return True


def runAquaduct(working_dir: str):
    os.chdir(working_dir)
    cmd = [
        "conda",
        "run -n aquaduct",
        "valve.py",
        "-c",
        "aquaduct_config.txt",
        "-t 1",
    ]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return -1, str(e)
    return 0, "all good"


def prepAquaduct(
    config: conf.JobConfig, workspace: fs.AdaptiveWorkplace, epoch_num: int
):
    logger = logging.getLogger(__name__)

    input_dirs = []
    for i in range(config.general.num_seeds):
        # get last epoch directory
        run_dir = workspace.get_files(
            workspace.get_files_by_tags([f"run_dir_epoch_{epoch_num}", f"run_seed_{i}"])
        )
        input_dirs.append(run_dir.abs_path)
    with mp.Pool(np.max([config.slurm_master.ncpus - 1, 1])) as p:
        results = p.map(runAquaduct, input_dirs)

    failed = [i for i,x in results if x[0]!=0]
    if len(failed)>0:
        for idx in failed:
            logger.error(f"aquaduct failed with error {results[idx][1]}")
    else:
        logger.info(f"aquaduct completed for epoch {epoch_num}")


def runStrip(topo, traj, mask):
    working_dir = os.path.join(topo.abs_path, os.pardir)
    os.chdir(working_dir)
    top = pt.load_topology(topo.abs_path)
    crd = pt.load(traj.abs_path, top=top)
    stripped_traj = crd.strip(mask=mask)
    stripped_parm = top.strip(mask=mask)
    straj = os.path.join(working_dir, "stripped.nc")
    sparm = os.path.join(working_dir, "stripped.parm7")
    pt.save(sparm, stripped_parm)
    pt.save(straj, stripped_traj)
    return sparm, straj, topo, traj


def prepStrip(config: conf.JobConfig, workspace: fs.AdaptiveWorkplace, epoch_num: int):
    logger = logging.getLogger(__name__)
    logger.info("strip started")
    # get topos and coords
    inputs = []
    for i in range(config.general.num_seeds):
        topo = workspace.get_files(
            workspace.get_files_by_tags(
                [f"run_epoch_{epoch_num}", f"seed_{i}", "topology"]
            )
        )
        traj = workspace.get_files(
            workspace.get_files_by_tags(
                [f"run_epoch_{epoch_num}", f"seed_{i}", "prod_traj"]
            )
        )
        #logger.info(f"strip args: {(topo, traj, config.aquaduct.post_run_strip_mask)}")
        inputs.append((topo, traj, config.aquaduct.post_run_strip_mask))
    with mp.Pool(np.max([config.slurm_master.ncpus - 1, 1])) as p:
        stripped_data = p.starmap(runStrip, inputs)
    for paths in stripped_data:
        workspace.add_file(paths[0], tags=paths[2].tags)
        workspace.add_file(paths[1], tags=paths[3].tags)
        workspace.unregister_file(paths[2])
        workspace.unregister_file(paths[3], remove=True)
    logger.info(
        f"Solvent selected by mask {config.aquaduct.post_run_strip_mask} was stripped from the trajectories"
    )


def tmpProteinProjection(traj):
    feats = []
    # here extract features using pytraj
    dihedrals = pt.multidihedral(
        traj=traj, dihedral_types="phi psi", resrange=range(11, 286)
    )
    # assing features to feats variable in way that feats is a list of lists (list for each feature)
    for dih in dihedrals:
        feats.append(dih.values)
    combined = np.column_stack(feats)
    # np.array is a expected return rows:frames, columns:features
    return combined


def uniq_sort(array):
    n = array.shape[0]
    if n == 1:
        return np.array([0])
    if n == 0:
        return np.array([])
    diff = array[:, None, :] - array[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_matrix, 0)
    avg_distances = np.sum(dist_matrix, axis=1) / (n - 1)
    sorted_idx = np.argsort(avg_distances)[::-1]
    return sorted_idx


def processSimulations(
    config: conf.JobConfig,
    log_path: str,
    workspace: fs.AdaptiveWorkplace,
    submitit_results,
    epoch_num: int,
):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    os.chdir(config.working_dir)

    if not validateEpoch(
        workspace, config, epoch_num, config.general.prod_num_frames, submitit_results
    ):
        return -1

    if config.aquaduct.run_aquaduct:
        prepAquaduct(config, workspace, epoch_num)

    if config.aquaduct.post_run_strip_mask is not None:
        prepStrip(config, workspace, epoch_num)

    # load projection function
    with open(
        workspace.get_files(workspace.get_files_by_tags("projection")).abs_path, "r"
    ) as f:
        exec(f.read(), globals())

    # for each seed in epoch
    for i in range(config.general.num_seeds):
        # load prod trajectory (which may be stripped)
        topo = workspace.get_files(
            workspace.get_files_by_tags(
                [f"run_epoch_{epoch_num}", f"seed_{i}", "topology"]
            )
        )
        crds = workspace.get_files(
            workspace.get_files_by_tags(
                [f"seed_{i}", f"run_epoch_{epoch_num}", "prod_traj"]
            )
        )
        traj = pt.load(crds.abs_path, top=topo.abs_path)
        # project trajectory
        projected_traj = projectTrajectory(traj)
        protein_dih_projection = tmpProteinProjection(traj)

        del traj
        # save timeseries
        run_dir = workspace.get_files(
            workspace.get_files_by_tags([f"run_dir_epoch_{epoch_num}", f"run_seed_{i}"])
        )
        arr_path = os.path.join(run_dir.abs_path, "prod_projection.npy")
        np.save(arr_path, projected_traj, allow_pickle=False)
        workspace.add_file(
            arr_path, tags=[f"seed_{i}", f"run_epoch_{epoch_num}", "prod_projection"]
        )
        arr_path = os.path.join(run_dir.abs_path, "dihedrals.npy")
        np.save(arr_path, protein_dih_projection, allow_pickle=False)
        workspace.add_file(
            arr_path,
            tags=[f"seed_{i}", f"run_epoch_{epoch_num}", "dihedral_projection"],
        )

    logger.info(f"Success in projection of epoch {epoch_num} trajectories")
    # load all projections and also dihedral projection
    featurized_trajs = []
    dihedral_trajs = {}
    for e in range(1, epoch_num + 1):
        for i in range(config.general.num_seeds):
            arr_path = workspace.get_files(
                workspace.get_files_by_tags(
                    [f"seed_{i}", f"run_epoch_{e}", "prod_projection"]
                )
            )
            array = np.load(arr_path.abs_path)
            parent_dir = workspace.get_files(
                workspace.get_files_by_tags([f"run_dir_epoch_{e}", f"run_seed_{i}"])
            ).abs_path
            featurized_trajs.append((parent_dir, array))
            arr_path = workspace.get_files(
                workspace.get_files_by_tags(
                    [f"seed_{i}", f"run_epoch_{e}", "dihedral_projection"]
                )
            )
            array = np.load(arr_path.abs_path)
            dihedral_trajs[parent_dir] = array
    logger.info(f"Loaded {len(featurized_trajs)} timeseries for model construction")

    if config.ligand_model.do_tica:
        tica = dt.decomposition.TICA(
            lagtime=config.ligand_model.ticalag, var_cutoff=0.95
        )
        processed_trajs = tica.fit_transform([x[1] for x in featurized_trajs])
        processed_trajs = [(x[0], y) for x, y in zip(featurized_trajs, processed_trajs)]
        logger.info(
            f"tICA with lagtime {config.ligand_model.ticalag} and {tica.dim} dims explained {tica.fetch_model().cumulative_kinetic_variance}"
        )

    # microstate clustering with kmeansminibatch
    concatenated_trajs = np.concatenate([x[1] for x in processed_trajs])
    if config.ligand_model.num_micro == -1:
        num_micro = getNumMicrostates(np.shape(concatenated_trajs)[0])
    else:
        num_micro = config.ligand_model.num_micro

    clusterer = dt.clustering.MiniBatchKMeans(
        num_micro, n_jobs=config.slurm_master.ncpus
    )
    logger.info(f"Discretized trajectories into {num_micro} microstates")
    concatenated_microstates = clusterer.fit_transform(concatenated_trajs)
    microstate_trajs = []
    lastpos = 0
    for traj in processed_trajs:
        fpos = lastpos + len(traj[1])
        microstate_trajs.append((traj[0], concatenated_microstates[lastpos:fpos]))
        lastpos = fpos

    counts = dt.markov.TransitionCountEstimator(
        lagtime=config.ligand_model.markov_lag, count_mode="sliding"
    )
    counts_model = counts.fit_fetch([x[1] for x in microstate_trajs])
    sets = counts_model.connected_sets()
    submodels_size = [len(x) for x in sets]
    logger.info("Microstate populations of connected sets:")
    for x in submodels_size:
        if x / num_micro * 100 > 5.0:
            logger.info(f"{x} : {x/num_micro * 100}%")
        else:
            pass
    msm = dt.markov.msm.MaximumLikelihoodMSM(
        allow_disconnected=False, use_lcc=False
    ).fit_fetch(counts_model.submodel_largest())
    # dump msm to pickle, no need to register
    dump_path = os.path.join(config.working_dir, f"epoch_{epoch_num}_MSM.dump")
    with open(dump_path, "wb") as fh:
        pickle.dump(msm, fh)

    num_macro = getNumMacrostates(config, [x[1] for x in microstate_trajs], num_micro)
    logger.info(f"Building MSM with {num_macro} metastable states")
    coarse_msm = msm.pcca(num_macro)
    micronum = len(msm.count_model.state_symbols)
    # dump msm coarse to pickle, no need to register
    dump_path = os.path.join(config.working_dir, f"epoch_{epoch_num}_coarse_MSM.dump")
    with open(dump_path, "wb") as fh:
        pickle.dump(coarse_msm, fh)

    logger.info(
        f"Largest connected submodel contains {micronum} microstates out of {num_micro} indentified"
    )
    logger.debug(f"len of assigment matrix {len(coarse_msm.assignments)}")

    res = np.zeros(num_macro)
    microvalue = np.ones(micronum)
    micro_mult = np.zeros(micronum)
    for i in range(len(microvalue)):
        macro = coarse_msm.assignments[i]
        membership_value = coarse_msm.memberships[i][macro]
        res[macro] = res[macro] + microvalue[i]
        # discriminate agains poorly defined microstates
        micro_mult[i] = membership_value

    nMicroPerMacro = res
    N = np.zeros(num_micro)
    for frame in concatenated_microstates:
        N[frame] = N[frame] + 1
    microvalue = N[msm.count_model.state_symbols]

    res = np.zeros(num_macro)
    for i in range(len(microvalue)):
        macro = coarse_msm.assignments[i]
        res[macro] = res[macro] + microvalue[i]
    try:
        p_i = 1 / res
    except ZeroDivisionError:
        p_i = 1 / (res + 0.000001)

    p_i = p_i / nMicroPerMacro
    respawn_weights = p_i[coarse_msm.assignments]

    labeled_statelist = []
    for traj in microstate_trajs:
        # (source, microstate, frame IS 0-indexed)
        labeled_statelist += [(traj[0], x, i) for i, x in enumerate(traj[1])]

    # restrict microstates to active set??

    # get spawn frames

    # possibly mapping of spawncounts to original microstate labels might be necessary
    active_set = msm.count_model.state_symbols
    memberships = coarse_msm.memberships
    logger.debug(f"memberships array: {np.shape(memberships)}")
    logger.info(
        f"reducing active set with {len(active_set)} microstates to core set with threshold of 0.5"
    )
    core_idx = get_assigned_microstates(memberships)[0]
    core_set = active_set[core_idx]
    prob = respawn_weights / np.sum(respawn_weights)
    core_prob = prob[core_idx]

    gen = np.random.default_rng(seed=config.general.os_enthropy_seed)
    spawncounts = gen.multinomial(config.general.num_seeds, core_prob)
    logger.info(f"resulting core set has {len(core_set)}")
    spawncounts_mapped = np.zeros(num_micro)
    spawncounts_mapped[core_set] = spawncounts
    assignments_mapped = np.ones(num_micro) * -1
    assignments_mapped[core_set] = coarse_msm.assignments[core_idx]

    frame_selection = []
    # list to be populated by (source, microstate, frame 0-indexed )
    stateidx = np.where(spawncounts_mapped > 0)[0]
    logger.info("Selecting respawn frames:")
    for state in stateidx:
        to_sample = int(spawncounts_mapped[state])
        microstate_tuples_idx = np.where(concatenated_microstates == state)[0]
        microstate_tuples = [labeled_statelist[x] for x in microstate_tuples_idx]
        dihedral_data = []
        for tup in microstate_tuples:
            dihs = dihedral_trajs[tup[0]][tup[2]]
            dihedral_data.append(dihs)
        dihedral_data = np.row_stack(dihedral_data)
        uniq_sorted_idx = uniq_sort(dihedral_data)
        if len(uniq_sorted_idx) >= to_sample:
            selected_idx = uniq_sorted_idx[:to_sample]
        else:
            logger.info(
                f"population of microstate {state} is too small to cover given respawn value, respawn conformation will repeat"
            )
            mult = np.ceil((to_sample / len(uniq_sorted_idx)))
            uniq_sorted_idx_mult = uniq_sorted_idx * mult
            selected_idx = uniq_sorted_idx_mult[:to_sample]
        for idx in selected_idx:
            frame = microstate_tuples[idx]
            logger.info(f"Selected frame {frame[2]} from simulation {frame[0]}")
            logger.info(
                f"representing microstate {frame[1]} assigned to macrostate {assignments_mapped[frame[1]]}"
            )
            frame_selection.append(frame)

    respawn_by_source_sim = {}
    for frame in frame_selection:
        if frame[0] not in respawn_by_source_sim:
            respawn_by_source_sim[frame[0]] = []
        respawn_by_source_sim[frame[0]].append(frame)

    os.chdir(config.working_dir)
    respawn_root = f"epoch_{epoch_num}_respawn"
    os.mkdir(respawn_root)
    workspace.add_file(respawn_root, tags=[f"epoch_{epoch_num}", "init_dir"])

    seed_num = 0
    for e in range(1, epoch_num + 1):
        for i in range(config.general.num_seeds):
            respawn_dir = workspace.get_files(
                workspace.get_files_by_tags([f"run_dir_epoch_{e}", f"run_seed_{i}"])
            )
            if respawn_dir.abs_path in respawn_by_source_sim:
                topo = workspace.get_files(
                    workspace.get_files_by_tags(
                        [f"run_epoch_{e}", f"seed_{i}", "topology"]
                    )
                )
                traj = workspace.get_files(
                    workspace.get_files_by_tags(
                        [f"run_epoch_{e}", f"seed_{i}", "prod_traj"]
                    )
                )
                loaded = pt.load(traj.abs_path, top=topo.abs_path)
                for frame in respawn_by_source_sim[respawn_dir.abs_path]:
                    print(frame)
                    core_name = (
                        respawn_dir.filename.split("_")[0]
                        + respawn_dir.filename.split("_")[-1]
                    )
                    dir_name = "_".join(
                        [
                            f"e{epoch_num+1}s{seed_num}",
                            core_name,
                            f"f{frame[2]}m{frame[1]}",
                        ]
                    )
                    dir_path = os.path.join(respawn_root, dir_name)
                    os.mkdir(dir_path)
                    workspace.add_file(
                        dir_path,
                        tags=[f"epoch_{epoch_num}_dir", f"seed_dir_{seed_num}"],
                    )
                    shutil.copy(topo.abs_path, dir_path)
                    workspace.add_file(
                        os.path.join(dir_path, topo.filename),
                        tags=["topology", f"seed_{seed_num}", f"epoch_{epoch_num}"],
                    )
                    crd_path = os.path.join(dir_path, "selected_respawn.rst")
                    pt.write_traj(crd_path, loaded, frame_indices=[frame[2]])
                    shutil.move(
                        os.path.join(dir_path, "selected_respawn.rst.1"), crd_path
                    )
                    workspace.add_file(
                        crd_path,
                        tags=[f"seed_{seed_num}", "coords", f"epoch_{epoch_num}"],
                    )
                    seed_num += 1

    if workspace.check_all_files():
        logger.critical(
            f"File incosistencies after seed preparation after run of epoch {epoch_num}"
        )
        return -1
    else:
        logger.info(f"post-processing of epoch {epoch_num} was completed successfuly")
        return workspace
