from engine.tools.convert_config_to_dataloader import \
    convert_config_to_dataloader
from engine.data_preprocessor import MMGKDataPreProcessor


def main():
    dl, ds = convert_config_to_dataloader(
        'engine/datasets/test/train_simaug_vct_actions.py', 'train'
    )
    preprocessor = MMGKDataPreProcessor()
    sample = next(iter(dl))
    sample = preprocessor(sample)
    print(sample)


if __name__ == '__main__':
    main()
