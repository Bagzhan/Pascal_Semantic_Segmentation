import segmentation_models_pytorch as smp

def get_model(num_classes=7):
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes
    )
    return model

