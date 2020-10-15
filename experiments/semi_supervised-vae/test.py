import torch
import pytorch_lightning as pl

from train import SSVAE


def main(checkpoint_path, hparams_path):

    model = SSVAE.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            hparams_file=hparams_path,
            map_location=None)

    trainer = pl.Trainer(
        progress_bar_refresh_rate=1,
        weights_summary='full',
        gpus=1 if torch.cuda.is_available() else 0,
        deterministic=True)

    # test (pass in the model)
    trainer.test(model)


if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])
