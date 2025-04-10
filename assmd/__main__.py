#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:51:42 2025

@author: igor
"""

import logging, argparse
import submitit as submit

from assmd import config as conf
from assmd import file_structure as fs
from assmd import sim_runner as sr
from assmd import adaptive_processing as ap
from assmd import workspace as ws



def main():    
    parser = argparse.ArgumentParser(
        description='Run adaptive molecular dynamics simulation with configuration file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file'
    )
    parser.add_argument(
        '-l', '--log',
        type=str,
        required=True,
        help='Path where log files will be written'
    )
    args = parser.parse_args()
    conf_file = args.config
    log_path = args.log
    
    logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()])
    
    
    logger = logging.getLogger(__name__)
    job_def=conf.loadConfig(conf_file)
    init_executor = submit.AutoExecutor(folder=job_def.slurm_log_dir)
    init_executor.update_parameters(timeout_min=10,
                                    tasks_per_node=1,
                                    nodes=1,
                                    slurm_partition=job_def.slurm_master.partition,
                                    slurm_job_name=job_def.slurm_master.name,
                                    cpus_per_task=2,
                                    slurm_account=job_def.slurm_master.account)
    init_job = init_executor.submit(ws.prepare_first_epoch,job_def)
    workspace = init_job.result()
    
    parsing_executor = submit.AutoExecutor(folder=job_def.slurm_log_dir)
    parsing_executor.update_parameters(timeout_min=job_def.slurm_master.time,
                                    tasks_per_node=1,
                                    nodes=1,
                                    slurm_partition=job_def.slurm_master.partition,
                                    slurm_job_name=job_def.slurm_master.name,
                                    cpus_per_task=job_def.slurm_master.ncpus,
                                    mem_gb=job_def.slurm_master.memory_gb,
                                    local_setup = job_def.slurm_master.setup_commands,
                                    slurm_account=job_def.slurm_master.account,
                                    slurm_setup = job_def.slurm_master.setup_commands)
    for epoch_num in range(1, job_def.adaptive.num_epoch+1):
        job_results = sr.launch_epoch(job_def, workspace, epoch_num)
        #print(len(workspace.files.keys()))
        job = parsing_executor.submit(ap.processSimulations, job_def, log_path, workspace, job_results, epoch_num)
        #print(len(workspace.files.keys()))
        workspace = job.result()
        #print(len(workspace.files.keys()))
        if epoch_num != job_def.adaptive.num_epoch:
            logger.info(f"preparing run files for epoch {epoch_num+1}")
            ws.prepare_epoch_run(job_def, workspace, epoch_num+1)                
    logger.info("ASSMD COMPLETED WITH HIGH CHANCE OF SUCCESS")
    
    
    
if __name__=="__main__":
    main()