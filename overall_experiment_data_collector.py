import argparse
from logger import NeptuneLogger

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_number', default=0, type=int, help='Which experiment number do you want to collect data for 1(A>A),2(A>X),3(A&B>A&B) or 4(A&B>X)')
parser.add_argument('--experiment_name', default="", type=str, help='What shall we call the overall experiment?')
parser.add_argument('--neptune_username', default="", type=str, help=' (Optional) For outputting training/eval metrics to neptune.ai. Valid neptune username. Not your neptune.ai API key must also be stored as $NEPTUNE_API_TOKEN environment variable.')
args = parser.parse_args()

logger = NeptuneLogger(args.neptune_username)
logger.log_output_files(experiment_name = args.experiment_name, experiment_number=args.experiment_number)
