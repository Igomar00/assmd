#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:07:43 2025
checkpointer system for assmd

@author: igor
"""
import sys, time, pickle, logging, os
import submitit as submit
from assmd import config as conf
from assmd import file_structure as fs
from assmd import adaptive_processing as ap
from assmd import workspace as ws
from typing import Literal

class Checkpoint():
    def __init__(self, log_path:str, config:conf.JobConfig, workspace:fs.AdaptiveWorkplace, epoch_num:int, stage:Literal["presim", "postsim", "processed"], submit_results=None):
        if workspace.check_all_files():
            print(f"Cant create checkpoint with corrupted workspace")
            sys.exit(1)
        self.time = time.asctime()
        self.log = log_path
        self.config = config
        self.epoch_num = epoch_num
        self.stage = stage
        self.workspace = workspace
        self.fname = f"{epoch_num}_{stage}.chkpoint"
        if self.stage=="postsim" and submit_results is not None:
            self.submitit_results = submit_results
    
    def save(self):
        path = os.path.join(self.config.working_dir, self.fname)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, pickle.HIGHEST_PROTOCOL)

    def load(filepath):
        with open(filepath, 'rb') as fh:
            chpoint = pickle.load(fh)
        return chpoint
    
def resume_from_checkpoint(chkpoint:Checkpoint):
    logger=logging.getLogger(__name__)
    config = chkpoint.config
    epoch_num = chkpoint.epoch_num
    stage = chkpoint.stage
    if epoch_num == config.general.num_epoch and stage=="processed":
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
        parsing_executor.update_parameters(timeout_min=config.slurm_master.time,
                                    tasks_per_node=1,
                                    nodes=1,
                                    slurm_partition=config.slurm_master.partition,
                                    slurm_job_name=config.slurm_master.name,
                                    cpus_per_task=config.slurm_master.ncpus,
                                    mem_gb=config.slurm_master.memory_gb,
                                    local_setup = config.slurm_master.setup_commands,
                                    slurm_account=config.slurm_master.account,
                                    slurm_setup = config.slurm_master.setup_commands)
        job = parsing_executor.submit(ap.processSimulations, config, chkpoint.log, workspace, chkpoint.submitit_results, epoch_num)
        workspace = job.result()
        stage = "processed"
    if stage=="processed":
        ws.prepare_epoch_run(config, workspace, epoch_num+1)
        return epoch_num+1, config, workspace

        