from data.types import DataLoaderTypes



def build_model(config):
    if config.model =='adabins':
        from .unet_adaptive_bins import UnetAdaptiveBins
        return UnetAdaptiveBins.build_from_config(config)

    if config.model == 'unet':
        from .unet_adaptive_bins import Unet
        return Unet.build_from_config(config)
    
    if config.model == 'localbins':
        from .localbins import UnetLocalBins
        return UnetLocalBins.build_from_config(config)

    # other models
    

    raise ValueError("Unknown model %s" % config.model)