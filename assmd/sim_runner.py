#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 01:19:46 2025

@author: igor
"""
import os, logging, subprocess, time, errno
import submitit as submit
from typing import List, Dict, Tuple

from assmd import file_structure as fs
from assmd import config as conf
from assmd import sim_api as sa

def launch_epoch(config:conf.JobConfig, workspace:fs.AdaptiveWorkplace, epoch_num)->list[Tuple[int, str]]:
    logger = logging.getLogger(__name__)
    logger.info(f"Starting simulations for epoch {epoch_num}")
    work_files = []
    for i in range(config.general.num_seeds):
        rdir = workspace.get_files(workspace.get_files_by_tags([f"run_dir_epoch_{epoch_num}", f"run_seed_{i}"]))
        topo = workspace.get_files(workspace.get_files_by_tags([f"run_epoch_{epoch_num}", f"seed_{i}", "topology"])).filename
        coords = workspace.get_files(workspace.get_files_by_tags([f"run_epoch_{epoch_num}", f"seed_{i}", "coords"])).filename
        work_files.append((rdir,topo,coords))

    node_executor = submit.AutoExecutor(folder = config.slurm_log_dir)
    node_executor.update_parameters(timeout_min=config.slurm_node.time,
                                    slurm_partition=config.slurm_node.partition,
                                    slurm_job_name=config.slurm_node.name,
                                    cpus_per_task=config.slurm_node.ncpus,
                                    mem_gb=config.slurm_node.memory_gb,
                                    slurm_gres=config.slurm_node.gres,
                                    slurm_account=config.slurm_node.account,
                                    slurm_qos=config.slurm_node.qos,
                                    slurm_array_parallelism=config.general.max_concurent,
                                    slurm_setup = config.slurm_node.setup_commands)
    outputs = []
    if config.general.pre_epoch_heating:
        n_work_files = []
        jobs = []
        with node_executor.batch():
            for wfiles in work_files:
                job = node_executor.submit(sa.launch_heating, wfiles[0], wfiles[1], wfiles[2])
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
                    result = jobs[idx].result()
                    if result[0] == -1 and not resubmitted[idx]:
                        # Resubmit failed job
                        logger.info(f"Resubmitting failed heating job for seed {idx} in epoch {epoch_num}")
                        new_job = node_executor.submit(sa.launch_heating, work_files[idx][0], work_files[idx][1], work_files[idx][2])
                        jobs[idx] = new_job
                        resubmitted[idx] = True
                        pending_jobs.append(idx)
        
        # Collect all results in original order
        outputs = [jobs[i].result() for i in range(len(jobs))]
        work_files = n_work_files
        
    if config.general.pre_epoch_equil:
        n_work_files = []
        jobs = []
        with node_executor.batch():
            for wfiles in work_files:
                job = node_executor.submit(sa.launch_equil, wfiles[0], wfiles[1], wfiles[2])
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
                    result = jobs[idx].result()
                    if result[0] == -1 and not resubmitted[idx]:
                        # Resubmit failed job
                        logger.info(f"Resubmitting failed equilibration job for seed {idx} in epoch {epoch_num}")
                        new_job = node_executor.submit(sa.launch_equil, work_files[idx][0], work_files[idx][1], work_files[idx][2])
                        jobs[idx] = new_job
                        resubmitted[idx] = True
                        pending_jobs.append(idx)
        
        # Collect all results in original order
        outputs += [jobs[i].result() for i in range(len(jobs))]
        work_files = n_work_files
    
    # Production run
    jobs = []
    with node_executor.batch():
        for wfiles in work_files:
            job = node_executor.submit(sa.launch_prod, wfiles[0], wfiles[1], wfiles[2])
            jobs.append(job)
    
    # Track resubmitted jobs
    resubmitted = [False] * len(jobs)
    pending_jobs = list(range(len(jobs)))
    
    while pending_jobs:
        time.sleep(config.general.update_rate)
        for idx in pending_jobs.copy():
            if jobs[idx].done():
                pending_jobs.remove(idx)
                result = jobs[idx].result()
                if result[0] == -1 and not resubmitted[idx]:
                    # Resubmit failed job
                    logger.info(f"Resubmitting failed production job for seed {idx} in epoch {epoch_num}")
                    new_job = node_executor.submit(sa.launch_prod, work_files[idx][0], work_files[idx][1], work_files[idx][2])
                    jobs[idx] = new_job
                    resubmitted[idx] = True
                    pending_jobs.append(idx)
    
    # Collect all results in original order
    outputs += [jobs[i].result() for i in range(len(jobs))]

    logger.info(f"Finished simulations for epoch {epoch_num}")
    return outputs