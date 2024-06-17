# Copyright (c) 2022-2024 Samuel Garcin
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT

from eval import *
from level_replay import MinigridDatasetSampler
from envs.multigrid.dataset import *

"""
Example usage:

python -m eval \
--env_name=MultiGrid-SixteenRooms-v0 \
--xpid=<xpid> \
--base_path="~/logs/dcd" \
--result_path="eval_results/"
--verbose
"""
def parse_args():
	parser = argparse.ArgumentParser(description='EvalDataset')


	parser.add_argument(
		'--datasets_root',
		type=str,
		default='datasets',
	)
	parser.add_argument(
		'--datasets',
		type=str,
		default='cave_escape_512_4ts_varp:test',
		help='Comma separated list of dataset names.')
	parser.add_argument(
		'--base_path',
		type=str,
		default='~/logs/dcd',
		help='Base path to experiment results directories.')
	parser.add_argument(
		'--xpid',
		type=str,
		default='latest',
		help='Experiment ID (result directory name) for evaluation.')
	parser.add_argument(
		'--prefix',
		type=str,
		default=None,
		help='Experiment ID prefix for evaluation (evaluate all matches).'
	)
	parser.add_argument(
		'--result_path',
		type=str,
		default=None,
		help='Relative path to evaluation results directory.')
	parser.add_argument(
		'--result_filename_prefix',
		type=str,
		default=None,
		help='Filename prefix.')
	parser.add_argument(
		'--max_seeds', 
		type=int, 
		default=None, 
		help='Maximum number of matched experiment IDs to evaluate.')
	parser.add_argument(
		'--num_processes',
		type=int,
		default=2,
		help='Number of CPU processes to use.')
	parser.add_argument(
		'--model_tar',
		type=str,
		default='model',
		help='Name of .tar to evaluate.')
	parser.add_argument(
		'--deterministic',
		type=str2bool, nargs='?', const=True, default=False,
		help="Evaluate policy greedily.")
	parser.add_argument(
		'--deterministic_start_state',
		type=str2bool, nargs='?', const=True, default=False,
		help="Environment initial state is deterministic given the level parameters.")
	parser.add_argument(
		'--verbose',
		type=str2bool, nargs='?', const=True, default=False,
		help="Show logging messages in stdout")
	parser.add_argument(
		'--render',
		type=str2bool, nargs='?', const=True, default=False,
		help="Render environment in first evaluation process to screen.")
	parser.add_argument(
		'--record_video',
		type=str2bool, nargs='?', const=True, default=False,
		help="Record video of first environment evaluation process.")

	return parser.parse_args()

def _get_dataset_env_name(dataset, deterministic_start_state=False):
	det_indicator = 'Deterministic' if deterministic_start_state else ''
	dataset2env = {
		'cave_escape_512_4ts_varp': f'MultiGrid-Dataset{det_indicator}Env-v0',
		'cave_escape_8_4ts_varp': f'MultiGrid-Dataset{det_indicator}Env-v0',
		'cave_escape_112_14ts_varp': f'MultiGrid-Dataset{det_indicator}Env-v0',
		'cave_escape_112_14ts_p50l_p5m': f'MultiGrid-Dataset{det_indicator}Env-v0',
		'cave_escape_112_14ts_varpl_p5m': f'MultiGrid-Dataset{det_indicator}Env-v0',
		'cave_escape_108_18ts_varp_dim45': f'MultiGrid-Dataset{det_indicator}S45Env-v0',
	}
	return dataset2env[dataset]


if __name__ == '__main__':
	os.environ["OMP_NUM_THREADS"] = "1"

	display = None
	if sys.platform.startswith('linux'):
		print('Setting up virtual display')

		import pyvirtualdisplay
		display = pyvirtualdisplay.Display(visible=0, size=(1400, 900), color_depth=24)
		display.start()

	args = DotDict(vars(parse_args()))

	# === Determine device ====
	device = 'cpu'

	# === Load checkpoint ===
	# Load meta.json into flags object
	base_path = os.path.expandvars(os.path.expanduser(args.base_path))

	xpids = [args.xpid]
	if args.prefix is not None:
		all_xpids = fnmatch.filter(os.listdir(base_path), f"{args.prefix}*")
		filter_re = re.compile('.*_[0-9]*$')
		xpids = [x for x in all_xpids if filter_re.match(x)]

	if args.model_tar == 'all':
		xpid_dir = os.path.join(base_path, xpids[0])
		checkpoint_files = [f.split('.')[0] for f in os.listdir(xpid_dir) if f.endswith('.tar')]
	else:
		checkpoint_files = args.model_tar.split(',')

	# Set up results management
	if args.result_path is None:
		args.result_path = os.path.join(base_path, 'evaluation')
	os.makedirs(args.result_path, exist_ok=True)

	for checkpoint_file in checkpoint_files:
		if args.result_filename_prefix is None:
			if args.prefix is not None:
				result_fname = args.prefix
			else:
				result_fname = args.xpid
		else:
			result_fname = args.result_filename_prefix
		result_fname = f"{result_fname}-{checkpoint_file}"
		result_fpath = os.path.join(args.result_path, result_fname)
		if os.path.exists(f'{result_fpath}.csv'):
			result_fpath = os.path.join(args.result_path, f'{result_fname}_redo')
		result_fpath = f'{result_fpath}.csv'

		csvout = open(result_fpath, 'w', newline='')
		csvwriter = csv.writer(csvout)

		env_results = defaultdict(list)

		datasets = args.datasets.split(',')
		num_datasets = len(datasets)

		if args.record_video:
			args.num_processes = 1

		num_seeds = 0
		for xpid in xpids:
			if args.max_seeds is not None and num_seeds >= args.max_seeds:
				break

			xpid_dir = os.path.join(base_path, xpid)
			meta_json_path = os.path.join(xpid_dir, 'meta.json')

			checkpoint_path = os.path.join(xpid_dir, f'{checkpoint_file}.tar')

			if os.path.exists(checkpoint_path):
				meta_json_file = open(meta_json_path)
				xpid_flags = DotDict(json.load(meta_json_file)['args'])

				make_fn = [lambda: Evaluator.make_env('MultiGrid-DatasetEnv-v0')]
				dummy_venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
				dummy_venv = Evaluator.wrap_venv(dummy_venv, env_name='MultiGrid-DatasetEnv-v0', device=device)

				# Load the agent
				agent = make_agent(name='agent', env=dummy_venv, args=xpid_flags, device=device)

				try:
					checkpoint = torch.load(checkpoint_path, map_location='cpu')
				except:
					continue

				if 'runner_state_dict' in checkpoint:
					agent.algo.actor_critic.load_state_dict(checkpoint['runner_state_dict']['agent_state_dict']['agent'])
				else:
					agent.algo.actor_critic.load_state_dict(checkpoint)

				num_seeds += 1

				# Evaluate environment batch in increments of chunk size
				for i in range(num_datasets):
					dataset_name, split = datasets[i].split(':')
					dataset_path = os.path.join(args.datasets_root, dataset_name)
					print(f'Evaluating {xpid} over {dataset_name}:{split}...')

					# Evaluate the model
					xpid_flags.update(args)
					xpid_flags.update({"use_skip": False})

					dataset_sampler = MinigridDatasetSampler(dataset_path, reload_level_encodings=True)
					if split == 'train':
						levels = dataset_sampler.train_levels
						labels = dataset_sampler.train_labels
					elif split == 'test':
						levels = dataset_sampler.test_levels
						labels = dataset_sampler.test_labels
					else:
						raise ValueError(f"Invalid split {args.split}")

					evaluator = Evaluator(
						env_names=[_get_dataset_env_name(dataset_name, args.deterministic_start_state)],
						num_processes=args.num_processes,
						num_episodes=len(levels),
						frame_stack=xpid_flags.frame_stack,
						grayscale=xpid_flags.grayscale,
						num_action_repeat=xpid_flags.num_action_repeat,
						use_global_critic=xpid_flags.use_global_critic,
						use_global_policy=xpid_flags.use_global_policy,
						device=device,
						levels=levels,
						labels=labels,
						next_level_on_reset='sequential'
					)

					stats = evaluator.evaluate(agent,
						deterministic=args.deterministic,
						show_progress=args.verbose,
						render=args.render,
						accumulator=None,
						eval_level_once=True)

					for k,v in stats.items():
						new_k = f'{datasets[i]}:{k}:{xpid}'
						env_results[new_k] += v

					evaluator.close()
			else:
				print(f'No model path {checkpoint_path}')

		output_results = {}
		for k,_ in env_results.items():
			if 'level_labels' in k:
				continue
			results = env_results[k]
			output_results[k] = f'{np.mean(results):.2f} +/- {np.std(results):.2f}'
			q1 = np.percentile(results, 25, interpolation='midpoint')
			q3 = np.percentile(results, 75, interpolation='midpoint')
			median = np.median(results)
			output_results[f'iq_{k}'] = f'{q1:.2f}--{median:.2f}--{q3:.2f}'
			print(f"{k}: {output_results[k]}")
		HumanOutputFormat(sys.stdout).writekvs(output_results)

		for k,v in env_results.items():
			row = [k,] + v
			csvwriter.writerow(row)

	if display:
		display.stop()
