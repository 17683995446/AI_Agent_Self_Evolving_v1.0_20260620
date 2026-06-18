#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Master script to write Part 2 (Phases 21-34) of the technical white paper.
Runs all three sub-scripts in sequence."""

import subprocess
import sys
import os

def run_script(script_path):
    """Run a Python script and check its return code."""
    print(f"Running: {script_path}")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR running {script_path}:")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout.strip())
    print(f"Completed: {script_path}\n")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts = [
        os.path.join(base_dir, "write_part2_part1.py"),
        os.path.join(base_dir, "write_part2_part2.py"),
        os.path.join(base_dir, "write_part2_part3.py"),
    ]
    
    for script in scripts:
        if not os.path.exists(script):
            print(f"ERROR: Script not found: {script}")
            sys.exit(1)
        run_script(script)
    
    print("All 14 phases (21-34) have been written successfully.")
    print("Output file: /workspace/自驱永续进化预测修正Agent_技术白皮书.md")

if __name__ == "__main__":
    main()