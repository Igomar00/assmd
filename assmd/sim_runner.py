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
    for i in range(config.adaptive.num_seeds):
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
                                    slurm_array_parallelism=config.adaptive.num_seeds,
                                    slurm_setup = config.slurm_node.setup_commands)
    outputs = []
    if config.init.pre_epoch_heating:
        n_work_files = []
        jobs = []
        with node_executor.batch():
            for wfiles in work_files:
                job = node_executor.submit(sa.launch_heating, wfiles[0], wfiles[1], wfiles[2])
                jobs.append(job)
                n_work_files.append((wfiles[0], wfiles[1], "heating_coords.rst"))
        working = True
        while working:
            time.sleep(config.adaptive.update_rate)
            working = not all([job.done() for job in jobs])
        outputs+=[job.result() for job in jobs]
        work_files = n_work_files
        
    if config.init.pre_epoch_equil:
        n_work_files = []
        jobs = []
        with node_executor.batch():
            for wfiles in work_files:
                job = node_executor.submit(sa.launch_equil, wfiles[0], wfiles[1], wfiles[2])
                jobs.append(job)
                n_work_files.append((wfiles[0], wfiles[1], "equil_coords.rst"))
        working = True
        while working:
            time.sleep(config.adaptive.update_rate)
            working = not all([job.done() for job in jobs])
        outputs+=[job.result() for job in jobs]
        work_files = n_work_files
    jobs = []
    with node_executor.batch():
        for wfiles in work_files:
            job = node_executor.submit(sa.launch_prod, wfiles[0], wfiles[1], wfiles[2])
            jobs.append(job)
    working = True
    while working:
        time.sleep(config.adaptive.update_rate)
        working = not all([job.done() for job in jobs])
    outputs+=[job.result() for job in jobs]

    logger.info(f"Finished simulations for epoch {epoch_num}")
    return outputs

