#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:26:07 2025

@author: igor
"""
import os
import sys
import shutil
import logging
import pytraj as pt

from assmd import file_structure as fs
from assmd import config as conf


def test_coords_and_project(tops, crds, proj_function):
    logger = logging.getLogger(__name__)
    try:
        with open(proj_function, "r") as f:
            exec(f.read(), globals())
    except SyntaxError as e:
        logger.critical(f"Syntax error in the code: {e}")
        logger.critical(f"Line number: {e.lineno}")
        logger.critical(f"Text: {e.text}")
        logger.critical(f"Offset: {e.offset}")
        sys.exit(1)
    except NameError as e:
        logger.critical(f"Name error (undefined variable): {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Other error occurred: {type(e).__name__}")
        logger.critical(f"Error message: {str(e)}")
        sys.exit(1)

    failed = False
    for i, seed in enumerate(zip(tops, crds)):
        result = None
        try:
            traj = pt.load(seed[1], top=seed[0])
        except Exception:
            logger.critical(f"failed to load init files for seed {i}")
            failed = True
        try:
            result = projectTrajectory(traj)
        except Exception as e:
            logger.critical(f"Projection function failed on seed {i} with error {e}")
            failed = True
        if result is not None:
            logger.debug(f"seed {i} resulted in projection")
            logger.debug(str(result))

    if failed:
        logger.critical("Input files verification failed - quitting")
        sys.exit(1)


def prepare_initial_workspace(config: conf.JobConfig) -> fs.AdaptiveWorkplace:
    logger = logging.getLogger(__name__)
    if not os.path.exists(config.working_dir):
        try:
            os.mkdir(config.working_dir)
        except Exception:
            logger.critical(
                f"Cant access or create working directory {config.working_dir}"
            )
            sys.exit(1)
    else:
        if len(os.listdir(config.working_dir)) > 0:
            logger.critical("working directory needs to be empty for initialization")
            sys.exit(1)
    os.chdir(config.working_dir)
    workspace = fs.AdaptiveWorkplace(config.working_dir)
    if (
        len(config.init.seed_paths) != 1
        and len(config.init.seed_paths) != config.general.num_seeds
    ):
        logger.critical(
            "number of seed crd must either be equal to 1 or to adaptive.num_seeds"
        )
        sys.exit(1)
    if len(config.init.seed_parm_paths) != len(config.init.seed_paths):
        logger.critical(
            "Number of topologies and coordinates in not in agreement - cant continue"
        )
        sys.exit(1)
    os.mkdir("epoch_0")
    workspace.add_file("epoch_0", tags=["epoch_0", "init_dir"])
    seed_info = open(os.path.join("epoch_0", "seed_assignment.txt"), "w")
    for i, seed in enumerate(
        zip(
            (
                config.init.seed_parm_paths
                if len(config.init.seed_parm_paths) > 1
                else config.init.seed_parm_paths * len(config.init.seed_paths)
            ),
            config.init.seed_paths,
        )
    ):
        seed_dir = os.path.join("epoch_0", f"seed_{i}")
        os.mkdir(seed_dir)
        workspace.add_file(seed_dir, tags=["epoch_0_dir", f"seed_dir_{i}"])
        extension = os.path.splitext(seed[0])[-1]
        parm_file = os.path.join(seed_dir, f"topology{extension}")
        try:
            shutil.copyfile(seed[0], parm_file)
        except Exception:
            logger.critical(f"Cant access file {seed[0]} for copying")
            sys.exit(1)
        workspace.add_file(
            parm_file, tags=["topology", f"seed_{i}", "epoch_0"], protected=True
        )
        extension = os.path.splitext(seed[1])[-1]
        coords_file = os.path.join(seed_dir, f"coords{extension}")
        try:
            shutil.copyfile(seed[1], coords_file)
        except Exception:
            logger.critical(f"Cant access file {seed[1]} for copying")
            sys.exit(1)
        workspace.add_file(
            coords_file, tags=[f"seed_{i}", "coords", "epoch_0"], protected=True
        )
        seed_info.write(
            f"seed_{i}\n\t{seed[0]} -> {parm_file}\n\t{seed[1]} -> {coords_file}\n"
        )
    seed_info.close()
    workspace.add_file(os.path.join("epoch_0", "seed_assignment.txt"), tags=["info"])

    try:
        shutil.copyfile(
            config.ligand_model.projection_function,
            os.path.join(config.working_dir, "projection.py"),
        )
    except Exception as e:
        logger.critical(f"cant copy projection function to working dir, {str(e)}")
        sys.exit(1)
    workspace.add_file(
        os.path.join(config.working_dir, "projection.py"), tags=["projection"]
    )

    if config.aquaduct.run_aquaduct:
        try:
            shutil.copyfile(
                config.aquaduct.config_file,
                os.path.join(config.working_dir, "aquaduct_config.txt"),
            )
        except Exception:
            logger.critical("cant copy aquaduct_config_file to the working dir")
            sys.exit(1)
        workspace.add_file(
            os.path.join(config.working_dir, "aquaduct_config.txt"),
            tags=["aqua", "config"],
        )

    crds, tops = [], []
    for i in range(config.general.num_seeds):
        tops.append(
            workspace.files[
                workspace.get_files_by_tags([f"seed_{i}", "topology"])
            ].abs_path
        )
        crds.append(
            workspace.files[
                workspace.get_files_by_tags([f"seed_{i}", "coords"])
            ].abs_path
        )
    test_coords_and_project(
        tops, crds, workspace.files[workspace.get_files_by_tags("projection")].abs_path
    )

    if len(workspace.get_files(workspace.get_files_by_tags("epoch_0_dir"))) == 1:
        for i in range(1, config.general.num_seeds):
            shutil.copytree(
                os.path.join("inputs", "seed_0"), os.path.join("inputs", f"seed_{i}")
            )
            for fid in workspace.files[workspace.get_files_by_tags("seed_0")]:
                to_register = os.path.join(
                    "inputs", f"seed_{i}", workspace.files[fid].filename
                )
                tags = workspace.files[fid].tags
                tags.remove("seed_0")
                tags.add(f"seed_{i}")
                workspace.add_file(to_register, tags=tags)
    try:
        shutil.copyfile(config.general.prod_config_path, "prod.in")
        workspace.add_file("prod.in", tags="prod_config")
        if config.general.pre_epoch_equil:
            shutil.copyfile(config.general.equil_config_path, "equil.in")
            workspace.add_file("equil.in", tags="equil_config")
        if config.general.pre_epoch_heating:
            shutil.copyfile(config.general.heating_config_path, "heating.in")
            workspace.add_file("heating.in", tags="heating_config")
    except Exception as e:
        logger.critical("unable to copy md config files")
        logger.critical(str(e))
        sys.exit(1)

    if workspace.check_all_files():
        logger.critical("Lack of integrity in files, quitting")
        sys.exit(1)
    return workspace


def prepare_epoch_run(
    config: conf.JobConfig, workspace: fs.AdaptiveWorkplace, epoch_num: int
):  # check protected
    os.chdir(config.working_dir)
    run_root = f"epoch_{epoch_num}_runs"
    os.mkdir(run_root)
    workspace.add_file(run_root, tags=[f"run_root_epoch_{epoch_num}"])
    for i in range(config.general.num_seeds):
        dname = os.path.join(
            run_root,
            workspace.get_files[
                workspace.get_files_by_tags(
                    [f"epoch_{epoch_num-1}_dir", f"seed_dir_{i}"]
                )
            ].filename,
        )
        os.mkdir(dname)
        print(dname)
        workspace.add_file(dname, tags=[f"run_dir_epoch_{epoch_num}", f"run_seed_{i}"])
        topo = workspace.get_files[
            workspace.get_files_by_tags(
                [f"epoch_{epoch_num-1}", f"seed_{i}", "topology"]
            )
        ]
        shutil.copy(topo.abs_path, dname)
        workspace.add_file(
            os.path.join(dname, topo.filename),
            tags=[f"run_epoch_{epoch_num}", f"seed_{i}", "topology"],
        )
        coords = workspace.get_files(
            workspace.get_files_by_tags([f"epoch_{epoch_num-1}", f"seed_{i}", "coords"])
        )
        if isinstance(coords, list):
            print([x.abs_path for x in coords])
        shutil.copy(coords.abs_path, dname)
        workspace.add_file(
            os.path.join(dname, coords.filename),
            tags=[f"run_epoch_{epoch_num}", f"seed_{i}", "coords"],
        )
        if config.general.pre_epoch_equil:
            conf = workspace.get_files[workspace.get_files_by_tags("equil_config")]
            shutil.copy(conf.abs_path, dname)
        if config.general.pre_epoch_heating:
            conf = workspace.get_files[workspace.get_files_by_tags("heating_config")]
            shutil.copy(conf.abs_path, dname)
        if config.aquaduct.run_aquaduct:
            conf = workspace.get_files[workspace.get_files_by_tags(["config", "aqua"])]
            shutil.copy(conf.abs_path, dname)
        conf = workspace.files[workspace.get_files_by_tags("prod_config")]
        shutil.copy(conf.abs_path, dname)

    if workspace.check_all_files():
        logger = logging.getLogger(__name__)
        logger.critical("File inconsistencies at rundir prepation")
        sys.exit(1)


def prepare_first_epoch(config: conf.JobConfig):
    workspace = prepare_initial_workspace(config)
    prepare_epoch_run(config, workspace, 1)
    return workspace
