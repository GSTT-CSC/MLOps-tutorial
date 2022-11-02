import pytorch_lightning as pl
from project.Network import Network
from project.DataModule import DataModule
import mlflow
from torch.cuda import is_available as cuda_available
import sys
import configparser
from pytorch_lightning.loggers import MLFlowLogger


def train(config):

    test_batch = 10  # set to >0 when you want a subest of the data for testing of size test_batch samples
    n_epochs = 5
    batch_size = 4

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
        mlf_logger = MLFlowLogger(
            experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=mlflow.active_run().info.run_id,
        )

        trainer = pl.Trainer(logger=mlf_logger,
                             precision=16 if cuda_available() else 32,
                             gpus=4 if cuda_available() else None,
                             max_epochs=n_epochs,
                             log_every_n_steps=1,
                             strategy="ddp" if cuda_available() else None,
                             accelerator="gpu" if cuda_available() else None,
                             )

        trainer.fit(net, dm)


if __name__ == '__main__':

    if len(sys.argv) > 0:
        config_path = sys.argv[1]
    else:
        config_path = './config/config.cfg'

    config = configparser.ConfigParser()
    config.read(config_path)
    train(config)