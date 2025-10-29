import errno
import subprocess
import os
import time
from typing import Tuple


def safe_chdir(directory):
    max_retries = 5
    retry_delay = 600
    for attempt in range(max_retries):
        try:
            os.chdir(directory)
            return True
        except OSError as e:
            if e.errno == errno.ESTALE:  # Stale file handle
                print(
                    f"Stale file handle on attempt {attempt+1}, retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                continue
            else:
                raise  # Re-raise if it's a different error

    # If we get here, all retries failed
    print(f"Failed to change to directory {directory} after {max_retries} attempts")
    return False


def launch_prod(working_directory, topology, coords) -> Tuple[int, str]:
    safe_chdir(working_directory)

    cmd = [
        "pmemd.cuda",
        "-O",
        "-i",
        "prod.in",
        "-p",
        topology,
        "-c",
        coords,
        "-o",
        "prod_log.out",
        "-r",
        "prod_coords.rst",
        "-x",
        "prod_traj.nc",
    ]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return -1, str(e)
    return 0, " ".join(cmd)


def launch_equil(working_directory, topology, coords) -> Tuple[int, str]:
    safe_chdir(working_directory)

    cmd = [
        "pmemd.cuda",
        "-O",
        "-i",
        "equil.in",
        "-p",
        topology,
        "-c",
        coords,
        "-o",
        "equil_log.out",
        "-r",
        "equil_coords.rst",
        "-x",
        "equil_traj.nc",
        "-ref",
        coords,
    ]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError:
        return -1, " ".join(cmd)
    return 0, " ".join(cmd)


def launch_heating(working_directory, topology, coords) -> Tuple[int, str]:
    safe_chdir(working_directory)

    cmd = [
        "pmemd.cuda",
        "-O",
        "-i",
        "heating.in",
        "-p",
        topology,
        "-c",
        coords,
        "-o",
        "heating_log.out",
        "-r",
        "heating_coords.rst",
        "-x",
        "heating_traj.nc",
        "-ref",
        coords,
    ]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError:
        return -1, " ".join(cmd)
    return 0, " ".join(cmd)


def launch_everything(working_directory, topology, coords) -> Tuple[int, str]:
    safe_chdir(working_directory)

    cmd = [
        "pmemd.cuda",
        "-O",
        "-i",
        "heating.in",
        "-p",
        topology,
        "-c",
        coords,
        "-o",
        "heating_log.out",
        "-r",
        "heating_coords.rst",
        "-x",
        "heating_traj.nc",
        "-ref",
        coords,
    ]

    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError:
        return -1, " ".join(cmd)

    cmd = [
        "pmemd.cuda",
        "-O",
        "-i",
        "equil.in",
        "-p",
        topology,
        "-c",
        "heating_coords.rst",
        "-o",
        "equil_log.out",
        "-r",
        "equil_coords.rst",
        "-x",
        "equil_traj.nc",
        "-ref",
        "heating_coords.rst",
    ]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError:
        return -1, " ".join(cmd)
    cmd = [
        "pmemd.cuda",
        "-O",
        "-i",
        "prod.in",
        "-p",
        topology,
        "-c",
        "equil_coords.rst",
        "-o",
        "prod_log.out",
        "-r",
        "prod_coords.rst",
        "-x",
        "prod_traj.nc",
    ]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return -1, str(e)
    return 0, " ".join(cmd)
