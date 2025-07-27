#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 09:42:43 2025
config file parsing for assmd
@author: igor
"""
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Literal
import os
import yaml


class SlurmConfig(BaseModel):
    partition: str
    time: int = Field(ge=1)
    setup_commands: List[str] = []
    memory_gb: Optional[int] = Field(None, ge=1)
    gres: Optional[str] = Field(default=None)
    output: str
    name: str
    ncpus: int = Field(ge=1, default=1)
    qos: str
    cluster: Literal["eagle", "labbit"]
    account: str


class MDConfig(BaseModel):
    seed_parm_paths: List[str] = []
    seed_paths: List[str] = []


class MSMConfig(BaseModel):
    projection_function: str
    do_tica: bool = Field(default=True)
    ticadim: Optional[int] = Field(ge=1)
    ticalag: Optional[int] = Field(ge=1, default=1)  # ticalag in frames!
    num_macro: Optional[int] = Field(ge=1, default=8)
    num_micro: Optional[int] = Field(ge=-1, default=-1)
    markov_lag: Optional[int] = Field(ge=1, default=1)


class General(BaseModel):
    prod_config_path: str
    pre_epoch_heating: bool
    heating_config_path: Optional[str] = None
    pre_epoch_equil: bool
    equil_config_path: Optional[str] = None
    prod_num_frames: int = Field(ge=1)
    num_seeds: int = Field(ge=1)
    num_epoch: int = Field(ge=1)
    max_concurent: Optional[int] = Field(ge=1)
    update_rate: Optional[int] = Field(ge=1, default=600)


class AquaDuctSetup(BaseModel):
    run_aquaduct: bool
    config_file: str = Field(default=None)
    post_run_strip_mask: str = Field(default=None)


class JobConfig(BaseModel):
    working_dir: str
    slurm_log_dir: str
    general: General
    slurm_node: SlurmConfig
    slurm_master: SlurmConfig
    init: MDConfig
    ligand_model: MSMConfig
    aquaduct: AquaDuctSetup


def loadConfig(config_path: str) -> JobConfig:  # Claude Sonnet 3.5
    """
    Load and validate configuration from a YAML file.    counts = dt.markov.TransitionCountEstimator(lagtime, count_mode)


    Args:
        config_path: Path to the YAML configuration file

    Returns:
        JobConfig: Validated configuration object

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
        ValidationError: If the configuration doesn't match the expected schema
        KeyError: If required sections are missing in the configuration
    """
    try:
        # Check if file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Read and parse YAML
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Check for top-level job key
        if "job" not in config_dict:
            raise KeyError("Configuration must have a top-level 'job' key")

        # Create and validate JobConfig object
        job_config = JobConfig(**config_dict["job"])

        return job_config

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")
    except ValidationError as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading configuration: {e}")
