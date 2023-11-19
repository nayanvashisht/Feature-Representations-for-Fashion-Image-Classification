import argparse

def Parse_Arguments():
	# Argument Parser
	parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


	# Mode
	parser.add_argument('--train', action='store_true', help='Training model on training dataset and simultaneouly validating on validation dataset.')
	parser.add_argument('--evaluate', action='store_true', help='Evaluating model on validation dataset.')


	# Path
	parser.add_argument('--main_path', default='', type=str, help='Path to main.py or train.py.')
	parser.add_argument('--resume_ckpt_path', default=None, type=str, help='Path to checkpoints to resume training. Note: The model will be save according to the main-path, dataset-name and model-name.')
	

	# Training Parameters
	parser.add_argument('--epochs', default=100, type=int, help='No.of total epochs for training. (default: 300)')
	parser.add_argument('--batch_size', default=16, type=int, help='Mini-batch size during training. This is the total batch size of all GPUs on the current node when using  Distributed Data Parallel Strategy. (default: 16)')

	
	# Distributed Training Parameters
	parser.add_argument('--num_nodes', default=1, type=int, help='Number of nodes for distributed training (default: 1).')
	parser.add_argument('--gpu', default=1, type=int, help='Number of GPUs per nodes for distributed training (default: 1).')

	return parser.parse_args()