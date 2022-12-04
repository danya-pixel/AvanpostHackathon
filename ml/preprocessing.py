import albumentations as albu
from albumentations.pytorch import ToTensor


def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms():
    result = [
        albu.RandomRotate90(),
        albu.Cutout(max_h_size=2, max_w_size=2),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        # albu.GridDistortion(p=0.2),
        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=10, p=0.3),
        albu.Blur(blur_limit=1),
        albu.HorizontalFlip(p=0.3),
    ]

    return result


def resize_transforms(image_size=224):
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose(
        [
            albu.SmallestMaxSize(pre_size, p=1),
            albu.RandomCrop(image_size, image_size, p=1),
        ]
    )
    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])
    result = [albu.OneOf([random_crop, rescale], p=1)]

    return result


def post_transforms():
    return [albu.Normalize(), ToTensor()]


def compose(transforms_to_compose):
    result = albu.Compose(
        [item for sublist in transforms_to_compose for item in sublist]
    )
    return result


def get_train_transforms():
    return compose([resize_transforms(), hard_transforms(), post_transforms()])


def get_show_transforms():
    return compose([resize_transforms(), hard_transforms()])


def get_test_transforms():
    return compose([pre_transforms(), post_transforms()])