from os.path import dirname, join

from ml_logger import RUN, instr
from termcolor import colored

assert instr  # single-entry for the instrumentation thunk factory
RUN.project = "ffn-code-release"  # Specify the project name
# RUN.job_name = "{RUN.count}"
RUN.job_name += "/{job_counter}"
RUN.prefix = "{project}/{project}/{username}/{now:%Y/%m-%d}/{file_stem}/{job_name}"
RUN.script_root = dirname(__file__)  # specify that this is the script root.
print(colored('set', 'blue'), colored("RUN.script_root", "yellow"), colored('to', 'blue'),
      RUN.script_root)
