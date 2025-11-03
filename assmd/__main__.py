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
    exclusive.add_argument(
        "-R",
        "--recover",
        type=str,
        help="Path to existing working directory to recover and create checkpoints from",
    )
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        required=True,
        help="Path where log files will be written",
    )
    parser.add_argument(
        "--config-for-recovery",
        type=str,
        help="Path to config file (required when using --recover mode)",
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

    elif args.recover:
        if not args.config_for_recovery:
            logger.critical(
                "--config-for-recovery is required when using --recover mode"
            )
            sys.exit(1)

        working_dir = args.recover
        conf_file = args.config_for_recovery

        logger.info(f"RECOVERY MODE: Reconstructing workspace from {working_dir}")
        job_def = conf.loadConfig(conf_file)

        # Attempt to recover workspace and determine the stage
        workspace, start_epoch, stage, submitit_results = chk.recover_workspace(
            working_dir, job_def, logger
        )

        if workspace is None:
            logger.critical("Failed to recover workspace from directory")
            sys.exit(1)

        logger.info(
            f"Successfully recovered workspace. Epoch {start_epoch}, Stage: {stage}"
        )

        # Handle postsim recovery - need to run adaptive processing immediately
        if stage == "postsim":
            logger.info(
                f"Running adaptive processing for epoch {start_epoch} (recovered from postsim)"
            )

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

            job = parsing_executor.submit(
                ap.processSimulations,
                job_def,
                log_path,
                workspace,
                submitit_results,
                start_epoch,
            )
            workspace = job.result()

            # Create processed checkpoint
            chkpoint = chk.Checkpoint(
                log_path, job_def, workspace, start_epoch, "processed"
            )
            chkpoint.save()
            logger.info(f"Adaptive processing complete for epoch {start_epoch}")

            # Prepare for next epoch if not at the end
            if start_epoch < job_def.general.num_epoch:
                logger.info(f"Preparing run files for epoch {start_epoch + 1}")
                ws.prepare_epoch_run(job_def, workspace, start_epoch + 1)

            # Now continue with the next epoch
            start_epoch = start_epoch + 1

        elif stage == "processed":
            # Need to prepare the next epoch's run files
            if start_epoch < job_def.general.num_epoch:
                logger.info(f"Preparing run files for epoch {start_epoch + 1}")
                ws.prepare_epoch_run(job_def, workspace, start_epoch + 1)
                start_epoch = start_epoch + 1
            else:
                logger.info("All epochs already completed")
                sys.exit(0)
        # else: stage == "presim", start_epoch is already set correctly

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
