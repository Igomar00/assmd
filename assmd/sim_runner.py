#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 01:19:46 2025

@author: igor
"""
import logging
import time
import os
import submitit as submit
from typing import Tuple

from assmd import file_structure as fs
from assmd import config as conf
from assmd import sim_api as sa

def check_file_exists(directory: str, filename: str) -> bool:
    """
    Check if a file exists in the given directory and has non-zero size.
    
    Args:
        directory (str): Directory path to check
        filename (str): Name of the file to check
    
    Returns:
        bool: True if file exists and has size > 0, False otherwise
    """
    try:
        file_path = os.path.join(directory, filename)
        if not os.path.exists(file_path):
            return False
        return os.path.getsize(file_path) > 0
    except (OSError, IOError):
        return False

def check_log_for_string(directory: str, log_filename: str, search_string: str, 
                        max_lines: int = 100) -> bool:
    """
    Check if a string exists in the last N lines of a log file.
    
    Args:
        directory (str): Directory containing the log file
        log_filename (str): Name of the log file
        search_string (str): String to search for
        max_lines (int): Maximum number of lines to check from end of file
    
    Returns:
        bool: True if string found in last max_lines, False otherwise
    """
    try:
        log_path = os.path.join(directory, log_filename)
        if not os.path.exists(log_path):
            return False
        
        with open(log_path, 'r') as f:
            # Read all lines
            lines = f.readlines()
            
            # Check only the last max_lines
            lines_to_check = lines[-max_lines:] if len(lines) > max_lines else lines
            
            # Search for the string
            for line in lines_to_check:
                if search_string in line:
                    return True
        
        return False
    
    except (OSError, IOError, UnicodeDecodeError):
        return False
    

def validate_simulation_outputs(directory: str, job_type: str) -> tuple[bool, str]:
    """
    Validate simulation outputs for a specific job type.
    
    Args:
        directory (str): Directory containing simulation outputs
        job_type (str): Type of job ('heating', 'equilibration', 'production')
    
    Returns:
        tuple[bool, str]: (success, message)
    """
    # Define expected files for each job type
    expected_files = {
        'heating': ['heating_coords.rst', "heating_traj.nc"],
        'equilibration': ['equil_coords.rst', 'equil_traj.nc'],
        'production': ['prod_traj.nc', "prod_coords.rst"]
    }
    
    # Define common log files to check
    log_files = [f"{job_type}_log.out"]
    
    # Check expected files exist
    files_to_check = expected_files.get(job_type, [])
    for filename in files_to_check:
        if not check_file_exists(directory, filename):
            return False, f"Missing or empty file: {filename}"
    
    # Check for completion string in any of the log files
    completion_found = False
    for log_file in log_files:
        if check_log_for_string(directory, log_file, "Total wall time"):
            completion_found = True
            break
    
    if not completion_found:
        return False, "No completion marker ('Total wall time') found in log files"
    
    return True, "All checks passed"

def launch_epoch(
    config: conf.JobConfig, workspace: fs.AdaptiveWorkplace, epoch_num
) -> list[Tuple[int, str]]:
    logger = logging.getLogger(__name__)
    logger.info(f"Starting simulations for epoch {epoch_num}")
    work_files = []
    for i in range(config.general.num_seeds):
        rdir = workspace.get_files(
            workspace.get_files_by_tags([f"run_dir_epoch_{epoch_num}", f"run_seed_{i}"])
        )
        topo = workspace.get_files(
            workspace.get_files_by_tags(
                [f"run_epoch_{epoch_num}", f"seed_{i}", "topology"]
            )
        ).filename
        coords = workspace.get_files(
            workspace.get_files_by_tags(
                [f"run_epoch_{epoch_num}", f"seed_{i}", "coords"]
            )
        ).filename
        work_files.append((rdir, topo, coords))

    node_executor = submit.AutoExecutor(folder=config.slurm_log_dir)
    node_executor.update_parameters(
        timeout_min=config.slurm_node.time,
        slurm_partition=config.slurm_node.partition,
        slurm_job_name=config.slurm_node.name,
        cpus_per_task=config.slurm_node.ncpus,
        mem_gb=config.slurm_node.memory_gb,
        slurm_gres=config.slurm_node.gres,
        slurm_account=config.slurm_node.account,
        slurm_qos=config.slurm_node.qos,
        slurm_array_parallelism=config.general.max_concurent,
        slurm_setup=config.slurm_node.setup_commands,
    )
    outputs = []
    if config.general.pre_epoch_heating:
        n_work_files = []
        jobs = []
        with node_executor.batch():
            for wfiles in work_files:
                job = node_executor.submit(
                    sa.launch_heating, [f.abs_path for f in wfiles[0]], wfiles[1], wfiles[2]
                )
                jobs.append(job)
                n_work_files.append((wfiles[0], wfiles[1], "heating_coords.rst"))

        # Track resubmitted jobs
        resubmitted = [False] * len(jobs)
        pending_jobs = list(range(len(jobs)))

        while pending_jobs:
            time.sleep(config.general.update_rate)
            for idx in pending_jobs.copy():
                if jobs[idx].done():
                    pending_jobs.remove(idx)
                    try:
                        result = jobs[idx].result()
                    except Exception as e:
                        logger.error(f"Job {idx} threw exception: {e}")
                        result = (-1, f"Exception: {str(e)}")  # Treat as failed job
                    status, desc = validate_simulation_outputs(work_files[idx][0].abs_path, "heating")
                    if not status or result[0]!=0:
                        if not resubmitted[idx]:
                            # Resubmit failed job
                            logger.info(
                                f"Resubmitting failed heating job for seed {idx} in epoch {epoch_num}"
                            )
                            logger.info(
                                f"due to {desc}"
                            )
                            new_job = node_executor.submit(
                                sa.launch_heating,
                                work_files[idx][0].abs_path,
                                work_files[idx][1],
                                work_files[idx][2],
                            )
                            jobs[idx] = new_job
                            resubmitted[idx] = True
                            pending_jobs.append(idx)
                        else:
                            logger.error(f"job for seed {idx} was already resubmitted")

        # Collect all results in original order
        outputs = [jobs[i].result() for i in range(len(jobs))]
        work_files = n_work_files

    if config.general.pre_epoch_equil:
        n_work_files = []
        jobs = []
        with node_executor.batch():
            for wfiles in work_files:
                job = node_executor.submit(
                    sa.launch_equil, [f.abs_path for f in wfiles[0]], wfiles[1], wfiles[2]
                )
                jobs.append(job)
                n_work_files.append((wfiles[0], wfiles[1], "equil_coords.rst"))

        # Track resubmitted jobs
        resubmitted = [False] * len(jobs)
        pending_jobs = list(range(len(jobs)))

        while pending_jobs:
            time.sleep(config.general.update_rate)
            for idx in pending_jobs.copy():
                if jobs[idx].done():
                    pending_jobs.remove(idx)
                    try:
                        result = jobs[idx].result()
                    except Exception as e:
                        logger.error(f"Job {idx} threw exception: {e}")
                        result = (-1, f"Exception: {str(e)}")  # Treat as failed job
                    status, desc = validate_simulation_outputs(work_files[idx][0].abs_path, "equilibration")
                    if not status or result[0]!=0:
                        if not resubmitted[idx]:
                            # Resubmit failed job
                            logger.info(
                                f"Resubmitting failed equil job for seed {idx} in epoch {epoch_num}"
                            )
                            logger.info(
                                f"due to {desc}"
                            )
                            new_job = node_executor.submit(
                                sa.launch_equil,
                                work_files[idx][0].abs_path,
                                work_files[idx][1],
                                work_files[idx][2],
                            )
                            jobs[idx] = new_job
                            resubmitted[idx] = True
                            pending_jobs.append(idx)
                        else:
                            logger.error(f"job for seed {idx} was already resubmitted")


        # Collect all results in original order
        outputs += [jobs[i].result() for i in range(len(jobs))]
        work_files = n_work_files

    # Production run
    jobs = []
    with node_executor.batch():
        for wfiles in work_files:
            job = node_executor.submit(sa.launch_prod, [f.abs_path for f in wfiles[0]], wfiles[1], wfiles[2])
            jobs.append(job)

    # Track resubmitted jobs
    resubmitted = [False] * len(jobs)
    pending_jobs = list(range(len(jobs)))

    while pending_jobs:
        time.sleep(config.general.update_rate)
        for idx in pending_jobs.copy():
            if jobs[idx].done():
                pending_jobs.remove(idx)
                try:
                    result = jobs[idx].result()
                except Exception as e:
                    logger.error(f"Job {idx} threw exception: {e}")
                    result = (-1, f"Exception: {str(e)}")  # Treat as failed job
                status, desc = validate_simulation_outputs(work_files[idx][0].abs_path, "production")
                if not status or result[0]!=0:
                    if not resubmitted[idx]:
                        # Resubmit failed job
                        logger.info(
                            f"Resubmitting failed prod job for seed {idx} in epoch {epoch_num}"
                        )
                        logger.info(
                            f"due to {desc}"
                        )
                        new_job = node_executor.submit(
                            sa.launch_prod,
                            work_files[idx][0].abs_path,
                            work_files[idx][1],
                            work_files[idx][2],
                        )
                        jobs[idx] = new_job
                        resubmitted[idx] = True
                        pending_jobs.append(idx)
                    else:
                        logger.error(f"job for seed {idx} was already resubmitted")

    # Collect all results in original order
    outputs += [jobs[i].result() for i in range(len(jobs))]

    logger.info(f"Finished simulations for epoch {epoch_num}")
    return outputs
