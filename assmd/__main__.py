#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:51:42 2025

@author: igor
"""

import sys
import logging
import argparse
import submitit as submit

from assmd import config as conf
from assmd import sim_runner as sr
from assmd import adaptive_processing as ap
from assmd import workspace as ws
from assmd import checkpointer as chk


def main():
    print(sys.argv)
    parser = argparse.ArgumentParser(
        description="Run adaptive molecular dynamics simulation with configuration file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    exclusive = parser.add_mutually_exclusive_group(required=True)
    exclusive.add_argument(
        "-c", "--config", type=str, help="Path to the YAML configuration file"
    )
    exclusive.add_argument(
        "-r",
        "--restart",
        type=str,
        help="Path to the .chkpoint file from started protocol",
    )
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        required=True,
        help="Path where log files will be written",
    )
    args = parser.parse_args()
    log_path = args.log

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    
    if args.config:
        conf_file = args.config
        job_def = conf.loadConfig(conf_file)
        logger.info(f"USING OS ENTROPY VALUE: {job_def.general.os_enthropy_seed}")


        # normal operation
        init_executor = submit.AutoExecutor(folder=job_def.slurm_log_dir)
        init_executor.update_parameters(
            timeout_min=10,
            tasks_per_node=1,
            nodes=1,
            slurm_partition=job_def.slurm_master.partition,
            slurm_job_name=job_def.slurm_master.name,
            cpus_per_task=2,
            slurm_account=job_def.slurm_master.account,
        )
        init_job = init_executor.submit(ws.prepare_first_epoch, job_def)
        workspace = init_job.result()

        start_epoch = 1
        #
    elif args.restart:
        chk_file = args.restart
        chkpoint = chk.Checkpoint.load(chk_file)
        start_epoch, job_def, workspace = chk.resume_from_checkpoint(chkpoint)

    parsing_executor = submit.AutoExecutor(folder=job_def.slurm_log_dir)
    parsing_executor.update_parameters(
        timeout_min=job_def.slurm_master.time,
        tasks_per_node=1,
        nodes=1,
        slurm_partition=job_def.slurm_master.partition,
        slurm_job_name=job_def.slurm_master.name,
        cpus_per_task=job_def.slurm_master.ncpus,
        mem_gb=job_def.slurm_master.memory_gb,
        local_setup=job_def.slurm_master.setup_commands,
        slurm_account=job_def.slurm_master.account,
        slurm_setup=job_def.slurm_master.setup_commands,
    )

    for epoch_num in range(start_epoch, job_def.general.num_epoch + 1):
        chkpoint = chk.Checkpoint(log_path, job_def, workspace, epoch_num, "presim")
        chkpoint.save()
        job_results = sr.launch_epoch(job_def, workspace, epoch_num)
        chkpoint = chk.Checkpoint(
            log_path, job_def, workspace, epoch_num, "postsim", job_results
        )
        chkpoint.save()
        # print(len(workspace.files.keys()))
        job = parsing_executor.submit(
            ap.processSimulations, job_def, log_path, workspace, job_results, epoch_num
        )
        # print(len(workspace.files.keys()))
        workspace = job.result()
        chkpoint = chk.Checkpoint(log_path, job_def, workspace, epoch_num, "processed")
        chkpoint.save()
        # print(len(workspace.files.keys()))
        if epoch_num != job_def.general.num_epoch:
            logger.info(f"preparing run files for epoch {epoch_num+1}")
            ws.prepare_epoch_run(job_def, workspace, epoch_num + 1)
    logger.info("ASSMD COMPLETED WITH HIGH CHANCE OF SUCCESS")


if __name__ == "__main__":
    main()
