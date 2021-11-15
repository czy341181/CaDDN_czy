import torchvision

from .ddn_template import DDNTemplate


class DDNDeepLabV3(DDNTemplate):

    def __init__(self, backbone_name, feat_extract_layer, num_classes, pretrained_path=None, aux_loss=None):
        """
        Initializes DDNDeepLabV3 model
        Args:
            backbone_name [str]: ResNet Backbone Name
        """
        if backbone_name == "ResNet50":
            constructor = torchvision.models.segmentation.deeplabv3_resnet50
        elif backbone_name == "ResNet101":
            constructor = torchvision.models.segmentation.deeplabv3_resnet101
        else:
            raise NotImplementedError

        super().__init__(constructor=constructor, feat_extract_layer=feat_extract_layer, num_classes=num_classes, pretrained_path=pretrained_path, aux_loss=aux_loss)
