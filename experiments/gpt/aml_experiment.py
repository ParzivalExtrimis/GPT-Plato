from uuid import uuid4
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.runconfig import ComputeTarget
from azureml.core.dataset import Dataset
import mlflow
import os

import git

#get global workspace
workspace = Workspace.from_config()
mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())

#prepare data
input_path = os.path.join('experiments','gpt','data.txt')
datastore = workspace.get_default_datastore()
print(datastore.name, datastore.datastore_type, datastore.account_name, datastore.container_name)

datastore.upload_files([input_path], target_path='sam_harris_dataset', overwrite=True)
ds_paths = [(datastore, 'sam_harris_dataset/')]
dataset = Dataset.File.from_files(path = ds_paths)
input_data = dataset.as_named_input('data').as_mount('/tmp/{}'.format(uuid4()))

#=========================================================================================================================================================

decorator = '\n=============================================================================================================================================\n'
#make scrpit config
prefix = git.Repo('.', search_parent_directories=True).working_tree_dir

if prefix is None:
    print('Git repo could not be initialized.')


gpu_compute_target = "gpu-compute"
registered_model_name = "plato-gpt-cuda-trained_model"
environment_name = "AzureML-ACPT-pytorch-1.12-py39-cuda11.6-gpu"
train_src_dir = prefix + '/experiments/gpt/src/'
os.makedirs(train_src_dir, exist_ok=True)

print(decorator, 'Input dataset: \n', input_data, decorator)
print(decorator, open(train_src_dir+'process.py', 'r').read(), decorator)

compute_target = ComputeTarget(workspace=workspace, name=gpu_compute_target)
experiment = Experiment(workspace=workspace, name="Default")
script_arguments = ['--data', input_data, '--registered_model_name', registered_model_name]

script_run_config = ScriptRunConfig(source_directory=train_src_dir,
                                   script='process.py',
                                   compute_target=compute_target,
                                   arguments=script_arguments)

# Set up the curated environment
environment = Environment.get(workspace=workspace, name=environment_name)
script_run_config.run_config.environment = environment

# Submit the script run config
run = experiment.submit(script_run_config)
run.display_name = 'CUDA-Training-Plato_GPT'

run.wait_for_completion(show_output=True)

#=========================================================================================================================================================

# Register the resulting model
model_save_path = os.path.join('outputs', experiment.name)
os.makedirs(model_save_path, exist_ok=True)

model = run.register_model(model_name=registered_model_name,
                           model_path=model_save_path,
                           tags={'experiment': experiment.name})

print('Model registered:', model.name, model.id)
