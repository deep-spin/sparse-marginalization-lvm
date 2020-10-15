import torch
import pytorch_lightning as pl

from train import SignalGame


def main(checkpoint_path, hparams_path):

    model = SignalGame.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            hparams_file=hparams_path,
            map_location=None)

    trainer = pl.Trainer(
        progress_bar_refresh_rate=1,
        weights_summary='full',
        limit_test_batches=10000//model.hparams.batch_size,
        gpus=1 if torch.cuda.is_available() else 0,
        deterministic=True)

    # test (pass in the model)
    trainer.test(model)


if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])
