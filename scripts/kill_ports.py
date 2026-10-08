"""Utility script to free ports 8000 (API) and 8501 (Dashboard) on Windows / Unix."""

import os
import subprocess
import sys


def free_port(port: int) -> None:
    if sys.platform == "win32":
        try:
            # Query netstat for active port
            out = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True, text=True)
            for line in out.strip().splitlines():
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in line:
                    pid = parts[-1]
                    print(f"Terminating lingering process on port {port} (PID: {pid})...")
                    os.system(f"taskkill /PID {pid} /F >nul 2>&1")
        except subprocess.CalledProcessError:
            print(f"Port {port} is free.")
    else:
        try:
            out = subprocess.check_output(f"lsof -t -i:{port}", shell=True, text=True)
            for pid in out.strip().splitlines():
                if pid:
                    os.system(f"kill -9 {pid} 2>/dev/null")
                    print(f"Terminated process on port {port} (PID: {pid}).")
        except subprocess.CalledProcessError:
            print(f"Port {port} is free.")


if __name__ == "__main__":
    for p in [8000, 8501]:
        free_port(p)
    print("Done. Ports 8000 and 8501 are ready.")
