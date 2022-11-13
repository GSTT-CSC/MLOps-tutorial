import matplotlib.pyplot as plt
import torch
import itertools
import random
import mlflow
from mlops.utils.logger import logger
import numpy as np
from monai.inferers import sliding_window_inference


def plot_inference_test(net, dm, n_samples_plot=4):
    net.eval()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu:0")
    net.to(device)

    with torch.no_grad():
        dl = dm.val_dataloader()
        n_samples_total = len(dl)
        sample_idx = random.sample(range(n_samples_total), min(n_samples_total, n_samples_plot))

        if n_samples_total == 0:
            logger.warning(f'Unable to create preview figure: Dataloader {dl} is empty')
            return
        fig, axs = plt.subplots(len(sample_idx), 4, figsize=(int(5*len(sample_idx)), 20), dpi=80)

        for i, n in enumerate(sample_idx):
            k = int(np.floor(n / dl.batch_size))
            val_data = next(itertools.islice(dl, k, None))

            roi_size = (-1, -1, -1)
            sw_batch_size = 1
            val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, net)
            test_slice = random.randint(0, val_data["image"].shape[4])
            input_image = val_data["image"].cpu().numpy()[0, 0, :, :, test_slice]
            label_image = val_data["label"].cpu().numpy()[0, 0, :, :, test_slice]
            pred_image = np.array(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, test_slice])

            axs[i, 0].set_title(f"image {k} slice{test_slice}")
            axs[i, 0].imshow(input_image, cmap="gray")
            axs[i, 1].set_title(f"LABEL_TRUE slice{test_slice}")
            axs[i, 1].imshow(label_image)
            axs[i, 2].set_title(f"LABEL_PRED slice{test_slice}")
            axs[i, 2].imshow(pred_image)
            axs[i, 3].set_title(f"LABEL_PRED_DIFF slice{test_slice}")
            axs[i, 3].imshow(label_image - pred_image)

        for ax in axs.flat:
            ax.label_outer()

        plt.tight_layout()
        plt.show()
        plt.savefig('example_inference.png')
        mlflow.log_artifact('example_inference.png')

