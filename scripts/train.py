import configparser
import sys

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from torch.cuda import is_available as cuda_available

from project.DataModule import DataModule
from project.Network import Network
from project.util.visualise import plot_inference_test


def train(config):

    test_batch = 20  # set to > 0 when you want a subset of the data for testing of size test_batch samples
    max_epochs = 1
    batch_size = 4
    n_gpu = 4
    log_torchscript = True

    xnat_configuration = {'server': config['xnat']['SERVER'],
                          'user': config['xnat']['USER'],
                          'password': config['xnat']['PASSWORD'],
                          'project': config['xnat']['PROJECT'],
                          'verify': config.getboolean('xnat', 'VERIFY')}

    print('Creating Network and DataModule')
    dm = DataModule(xnat_configuration=xnat_configuration, batch_size=batch_size, test_batch=test_batch)
    net = Network(dropout=0.2)

    print('Starting logged run')
    mlflow.pytorch.autolog(log_models=False)
    with mlflow.start_run(run_name='training') as run:
        print(f'Artifact location: {mlflow.get_artifact_uri()}')
        mlf_logger = MLFlowLogger(
            experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=mlflow.active_run().info.run_id,
        )

        trainer = pl.Trainer(logger=mlf_logger,
                             auto_select_gpus=True,
                             precision=16 if cuda_available() else 32,
                             accelerator='gpu' if cuda_available() else None,
                             devices=n_gpu if cuda_available() else None,
                             max_epochs=max_epochs,
                             log_every_n_steps=1,
                             strategy="ddp" if cuda_available() else None,
                             )

        trainer.fit(net, dm)

        plot_inference_test(net, dm)
        if log_torchscript:
            scripted_model = net.to_torchscript(file_path='model.ts')
            mlflow.pytorch.log_model(scripted_model, "model")

        print(f'Artifact location: {mlflow.get_artifact_uri()}')


if __name__ == '__main__':

    if len(sys.argv) > 0:
        config_path = sys.argv[1]
    else:
        config_path = '../config/config.cfg'

    config = configparser.ConfigParser()
    config.read(config_path)
    train(config)
