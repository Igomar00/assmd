#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:07:43 2025
checkpointer system for assmd

@author: igor
"""
import sys
import time
import pickle
import logging
import os
import submitit as submit
from assmd import config as conf
from assmd import file_structure as fs
from assmd import adaptive_processing as ap
from assmd import workspace as ws
from typing import Literal, Tuple, Optional


class Checkpoint:
    def __init__(
        self,
        log_path: str,
        config: conf.JobConfig,
        workspace: fs.AdaptiveWorkplace,
        epoch_num: int,
        stage: Literal["presim", "postsim", "processed"],
        submit_results=None,
    ):
        if workspace.check_all_files():
            print("Cant create checkpoint with corrupted workspace")
            sys.exit(1)
        self.time = time.asctime()
        self.log = log_path
        self.config = config
        self.epoch_num = epoch_num
        self.stage = stage
        self.workspace = workspace
        self.fname = f"{epoch_num}_{stage}.chkpoint"
        if self.stage == "postsim" and submit_results is not None:
            self.submitit_results = submit_results

    def save(self):
        path = os.path.join(self.config.working_dir, self.fname)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, pickle.HIGHEST_PROTOCOL)

    def load(filepath):
        with open(filepath, "rb") as fh:
            chpoint = pickle.load(fh)
        return chpoint


def resume_from_checkpoint(chkpoint: Checkpoint):
    logger = logging.getLogger(__name__)
    config = chkpoint.config
    epoch_num = chkpoint.epoch_num
    stage = chkpoint.stage
    if epoch_num == config.general.num_epoch and stage == "processed":
        logger.critical("Target epoch reached, nothing to restart")
        sys.exit(1)

    workspace = chkpoint.workspace
    if workspace.check_all_files():
        logger.critical(f"File integrity fault in loaded checkpoint {chkpoint.fname}")
        sys.exit(1)

    if stage == "presim":
        return epoch_num, config, workspace

    if stage == "postsim":
        parsing_executor = submit.AutoExecutor(folder=config.slurm_log_dir)
        parsing_executor.update_parameters(
            timeout_min=config.slurm_master.time,
            tasks_per_node=1,
            nodes=1,
            slurm_partition=config.slurm_master.partition,
            slurm_job_name=config.slurm_master.name,
            cpus_per_task=config.slurm_master.ncpus,
            mem_gb=config.slurm_master.memory_gb,
            local_setup=config.slurm_master.setup_commands,
            slurm_account=config.slurm_master.account,
            slurm_setup=config.slurm_master.setup_commands,
        )
        job = parsing_executor.submit(
            ap.processSimulations,
            config,
            chkpoint.log,
            workspace,
            chkpoint.submitit_results,
            epoch_num,
        )
        workspace = job.result()
        stage = "processed"
    if stage == "processed":
        ws.prepare_epoch_run(config, workspace, epoch_num + 1)
        return epoch_num + 1, config, workspace


def detect_epoch_state(
    working_dir: str, num_expected_seeds: int, logger
) -> Tuple[int, str]:
    """
    Detect the last completed or partially completed epoch and its stage.

    Args:
        working_dir: Path to working directory
        num_expected_seeds: Expected number of seeds per epoch
        logger: Logger instance

    Returns:
        Tuple of (epoch_num, stage) where stage is 'processed', 'postsim', or 'presim'
        - 'processed': epoch fully completed with respawn directory created
        - 'postsim': simulations finished but adaptive processing not done/failed
        - 'presim': epoch started but simulations not complete
    """
    last_completed_epoch = 0
    last_stage = "processed"

    # Check for epoch_0 (initialization)
    if not os.path.exists(os.path.join(working_dir, "epoch_0")):
        logger.warning("epoch_0 directory not found - workspace may be incomplete")
        return 0, "presim"

    # Check for completed respawn epochs
    epoch_num = 1
    while True:
        respawn_dir = os.path.join(working_dir, f"epoch_{epoch_num}_respawn")
        runs_dir = os.path.join(working_dir, f"epoch_{epoch_num}_runs")

        # Case 1: Respawn directory exists - epoch was fully processed
        if os.path.exists(respawn_dir):
            seed_dirs = [
                d
                for d in os.listdir(respawn_dir)
                if os.path.isdir(os.path.join(respawn_dir, d))
            ]

            if len(seed_dirs) == num_expected_seeds:
                last_completed_epoch = epoch_num
                last_stage = "processed"
                epoch_num += 1
                continue
            else:
                logger.warning(
                    f"Epoch {epoch_num} respawn incomplete: found {len(seed_dirs)} seeds, expected {num_expected_seeds}"
                )
                break

        # Case 2: Runs directory exists but no respawn - check if sims are complete
        if os.path.exists(runs_dir):
            run_dirs = [
                d
                for d in os.listdir(runs_dir)
                if os.path.isdir(os.path.join(runs_dir, d))
            ]

            # Check for production trajectories in run directories
            completed_runs = 0
            for run_dir in run_dirs:
                prod_traj = os.path.join(runs_dir, run_dir, "prod_traj.nc")
                if os.path.exists(prod_traj) and os.path.getsize(prod_traj) > 0:
                    completed_runs += 1

            if completed_runs == num_expected_seeds:
                # Simulations complete but processing not done
                logger.info(
                    f"Epoch {epoch_num} simulations complete but adaptive processing missing"
                )
                return epoch_num, "postsim"
            else:
                logger.info(
                    f"Epoch {epoch_num} runs incomplete: {completed_runs}/{num_expected_seeds} completed"
                )
                return epoch_num, "presim"
        else:
            # No runs directory - we've checked all completed epochs
            break

    logger.info(f"Last fully processed epoch: {last_completed_epoch}")
    return last_completed_epoch, last_stage


def detect_last_completed_epoch(
    working_dir: str, num_expected_seeds: int, logger
) -> int:
    """
    Detect the last fully completed epoch (backward compatibility wrapper).

    Args:
        working_dir: Path to working directory
        num_expected_seeds: Expected number of seeds per epoch
        logger: Logger instance

    Returns:
        Last completed epoch number (0 if only initialization exists)
    """
    epoch_num, stage = detect_epoch_state(working_dir, num_expected_seeds, logger)
    if stage == "processed":
        return epoch_num
    else:
        # If in postsim or presim stage, the previous epoch was the last completed
        return max(0, epoch_num - 1)


def recover_workspace(
    working_dir: str, config: conf.JobConfig, logger
) -> Tuple[Optional[fs.AdaptiveWorkplace], int, str, list]:
    """
    Recover workspace structure from existing working directory.

    Args:
        working_dir: Path to working directory
        config: Job configuration
        logger: Logger instance

    Returns:
        Tuple of (workspace, next_epoch_to_run, stage, submitit_results) or (None, 0, None, None) on failure
        stage can be 'processed', 'postsim', or 'presim'
    """
    if not os.path.exists(working_dir):
        logger.critical(f"Working directory does not exist: {working_dir}")
        return None, 0, None, None

    logger.info(f"Starting workspace recovery from {working_dir}")

    # Create workspace object
    workspace = fs.AdaptiveWorkplace(working_dir)

    # Detect last completed epoch and its state
    epoch_num, stage = detect_epoch_state(working_dir, config.general.num_seeds, logger)

    logger.info(f"Detected epoch {epoch_num} in stage: {stage}")

    # Register projection files
    proj_file = os.path.join(working_dir, "projection.py")
    if os.path.exists(proj_file):
        workspace.add_file(proj_file, tags=["projection"])
        logger.debug("Registered projection.py")
    else:
        logger.warning("projection.py not found")

    protein_proj_file = os.path.join(working_dir, "projection_protein.py")
    if os.path.exists(protein_proj_file):
        workspace.add_file(protein_proj_file, tags=["protein_projection"])
        logger.debug("Registered projection_protein.py")
    else:
        logger.warning("projection_protein.py not found")

    # Register MD config files
    prod_in = os.path.join(working_dir, "prod.in")
    if os.path.exists(prod_in):
        workspace.add_file(prod_in, tags="prod_config")
    equil_in = os.path.join(working_dir, "equil.in")
    if config.general.pre_epoch_equil and os.path.exists(equil_in):
        workspace.add_file(equil_in, tags="equil_config")
    heating_in = os.path.join(working_dir, "heating.in")
    if config.general.pre_epoch_heating and os.path.exists(heating_in):
        workspace.add_file(heating_in, tags="heating_config")

    # Recover epoch_0 (initialization)
    logger.info("Recovering epoch_0 (initialization)")
    epoch_0_dir = os.path.join(working_dir, "epoch_0")
    if os.path.exists(epoch_0_dir):
        workspace.add_file(epoch_0_dir, tags=["epoch_0", "init_dir"])

        # Register seed directories and files
        for i in range(config.general.num_seeds):
            seed_dir = os.path.join(epoch_0_dir, f"seed_{i}")
            if os.path.exists(seed_dir):
                workspace.add_file(seed_dir, tags=["epoch_0_dir", f"seed_dir_{i}"])

                # Find and register topology file
                for fname in os.listdir(seed_dir):
                    fpath = os.path.join(seed_dir, fname)
                    if fname.startswith("topology"):
                        workspace.add_file(
                            fpath,
                            tags=["topology", f"seed_{i}", "epoch_0"],
                            protected=True,
                        )
                        logger.debug(f"Registered {fpath}")
                    elif fname.startswith("coords"):
                        workspace.add_file(
                            fpath,
                            tags=[f"seed_{i}", "coords", "epoch_0"],
                            protected=True,
                        )
                        logger.debug(f"Registered {fpath}")

    # Determine which epochs to recover based on stage
    if stage == "postsim":
        # Recover up to and including the epoch that needs processing
        epochs_to_recover = range(1, epoch_num + 1)
        last_completed = epoch_num - 1
    elif stage == "processed":
        # Recover all completed epochs
        epochs_to_recover = range(1, epoch_num + 1)
        last_completed = epoch_num
    else:  # presim
        # Recover completed epochs only
        epochs_to_recover = range(1, epoch_num)
        last_completed = max(0, epoch_num - 1)

    # Recover completed epochs (respawn directories)
    for e in range(1, last_completed + 1):
        logger.info(f"Recovering epoch {e} (processed)")

        # Recover respawn directory
        respawn_dir = os.path.join(working_dir, f"epoch_{e}_respawn")
        if os.path.exists(respawn_dir):
            workspace.add_file(respawn_dir, tags=[f"epoch_{e}", "init_dir"])

            # Register seed directories
            seed_dirs = sorted(
                [
                    d
                    for d in os.listdir(respawn_dir)
                    if os.path.isdir(os.path.join(respawn_dir, d))
                ]
            )

            for seed_idx, seed_dir_name in enumerate(seed_dirs):
                seed_dir_path = os.path.join(respawn_dir, seed_dir_name)
                workspace.add_file(
                    seed_dir_path, tags=[f"epoch_{e}_dir", f"seed_dir_{seed_idx}"]
                )

                # Register files in seed directory
                for fname in os.listdir(seed_dir_path):
                    fpath = os.path.join(seed_dir_path, fname)
                    if os.path.isfile(fpath):
                        if "topology" in fname or fname.endswith((".prmtop", ".parm7")):
                            workspace.add_file(
                                fpath,
                                tags=["topology", f"seed_{seed_idx}", f"epoch_{e}"],
                            )
                        elif (
                            "coord" in fname
                            or "respawn" in fname
                            or fname.endswith((".rst", ".rst7", ".inpcrd"))
                        ):
                            workspace.add_file(
                                fpath, tags=[f"seed_{seed_idx}", "coords", f"epoch_{e}"]
                            )

    # Recover run directories for all relevant epochs
    for e in epochs_to_recover:
        logger.info(f"Recovering epoch {e} run directory")
        runs_dir = os.path.join(working_dir, f"epoch_{e}_runs")
        if os.path.exists(runs_dir):
            workspace.add_file(runs_dir, tags=[f"run_root_epoch_{e}"])

            run_dirs = sorted(
                [
                    d
                    for d in os.listdir(runs_dir)
                    if os.path.isdir(os.path.join(runs_dir, d))
                ]
            )

            for seed_idx, run_dir_name in enumerate(run_dirs):
                run_dir_path = os.path.join(runs_dir, run_dir_name)
                workspace.add_file(
                    run_dir_path, tags=[f"run_dir_epoch_{e}", f"run_seed_{seed_idx}"]
                )

                # Register files in run directory
                for fname in os.listdir(run_dir_path):
                    fpath = os.path.join(run_dir_path, fname)
                    if os.path.isfile(fpath):
                        if "topology" in fname or fname.endswith((".prmtop", ".parm7")):
                            workspace.add_file(
                                fpath,
                                tags=[f"run_epoch_{e}", f"seed_{seed_idx}", "topology"],
                            )
                        elif fname == "prod_traj.nc":
                            workspace.add_file(
                                fpath,
                                tags=[
                                    f"seed_{seed_idx}",
                                    f"run_epoch_{e}",
                                    "prod_traj",
                                ],
                            )
                        elif fname == "prod_projection.npy":
                            workspace.add_file(
                                fpath,
                                tags=[
                                    f"seed_{seed_idx}",
                                    f"run_epoch_{e}",
                                    "prod_projection",
                                ],
                            )
                        elif fname == "dihedrals.npy":
                            workspace.add_file(
                                fpath,
                                tags=[
                                    f"seed_{seed_idx}",
                                    f"run_epoch_{e}",
                                    "dihedral_projection",
                                ],
                            )
                        # Register initial coords for the run
                        elif (
                            fname.startswith("coords")
                            or fname.startswith("heating_coords")
                            or fname.startswith("equil_coords")
                        ) and fname.endswith((".rst", ".rst7", ".inpcrd")):
                            # Only register the starting coords, not intermediate ones
                            if e == 1 and fname.startswith("coords"):
                                workspace.add_file(
                                    fpath,
                                    tags=[
                                        f"run_epoch_{e}",
                                        f"seed_{seed_idx}",
                                        "coords",
                                    ],
                                )
                            elif e > 1 and "respawn" in fname:
                                workspace.add_file(
                                    fpath,
                                    tags=[
                                        f"run_epoch_{e}",
                                        f"seed_{seed_idx}",
                                        "coords",
                                    ],
                                )

    # Check workspace integrity
    if workspace.check_all_files():
        logger.error(
            "Workspace integrity check failed - some protected files may have been modified"
        )
        # Don't fail completely, just warn

    logger.info(f"Workspace recovery complete. Registered {len(workspace.files)} files")

    # Create mock submitit_results if we're in postsim stage
    submitit_results = None
    if stage == "postsim":
        logger.info("Creating mock submitit results for postsim recovery")
        submitit_results = []
        for i in range(config.general.num_seeds):
            # Mock successful completion - adaptive_processing will validate the files
            submitit_results.append(
                (0, f"Mock result for seed {i} - recovered from filesystem")
            )

    # Determine next action based on stage
    if stage == "processed":
        # All processing complete, prepare for next epoch
        next_epoch = epoch_num + 1
        checkpoint = Checkpoint(
            log_path="recovery.log",
            config=config,
            workspace=workspace,
            epoch_num=epoch_num,
            stage="processed",
        )
        checkpoint.save()
        logger.info(f"Created checkpoint: {checkpoint.fname}")
        logger.info(f"Will prepare for epoch {next_epoch}")

    elif stage == "postsim":
        # Simulations done, need to run processing
        next_epoch = epoch_num
        checkpoint = Checkpoint(
            log_path="recovery.log",
            config=config,
            workspace=workspace,
            epoch_num=epoch_num,
            stage="postsim",
            submit_results=submitit_results,
        )
        checkpoint.save()
        logger.info(f"Created checkpoint: {checkpoint.fname}")
        logger.info(f"Will run adaptive processing for epoch {next_epoch}")

    else:  # presim
        # Need to run simulations
        next_epoch = epoch_num
        logger.info(f"Will run simulations for epoch {next_epoch}")

    return workspace, next_epoch, stage, submitit_results
