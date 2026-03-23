from torchvision import transforms


def get_train_transform(image_size: int = 160):
    """
    Simple preprocessing pipeline for training data.
    Kept intentionally lightweight for the interim submission.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def get_eval_transform(image_size: int = 160):
    """
    Simple preprocessing pipeline for validation/test data.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
