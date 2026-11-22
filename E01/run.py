"""Offline launch for manuscript E01."""
import runpy,sys
from pathlib import Path
sys.argv=[str(Path(__file__).parents[1]/"reproduce.py"),"--experiment","E01"]+sys.argv[1:]
runpy.run_path(sys.argv[0],run_name="__main__")
