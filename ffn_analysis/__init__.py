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

AWS_REGIONS = [
    "ap-northeast-1", "ap-northeast-2", "ap-south-1", "ap-southeast-1",
    "ap-southeast-2", "eu-central-1", "eu-west-1", "sa-east-1", "us-east-1",
    "us-east-2", "us-west-1", "us-west-2", ]

import yaml

with open(join(dirname(__file__), "../ec2_setup/ec2_image_ids.yml"), 'r') as stream:
    IMAGE_IDS = yaml.load(stream, Loader=yaml.BaseLoader)
