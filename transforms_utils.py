from torchvision import transforms


def get_train_transform(image_size: int = 160):
    """
    Simple preprocessing pipeline for training data.
    Kept intentionally lightweight for the interim submission.
    """
    #return transforms.Compose([
    #    transforms.Resize((image_size, image_size)),
    #    transforms.ToTensor(),
    #])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])


def get_eval_transform(image_size: int = 160):
    """
    Simple preprocessing pipeline for validation/test data.
    """
    #return transforms.Compose([
    #    transforms.Resize((image_size, image_size)),
    #    transforms.ToTensor(),
    #])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])
