class Config(dict):
    def __getattr__(self, key):
        try: return self[key]
        except KeyError: raise AttributeError(key)
    def __setattr__(self, key, value): self[key] = value

def get_trans_unet_config():
    return Config(hidden_size=768, classifier="seg", decoder_channels=(256,128,64,32),
                  n_skip=3, skip_channels=[512,256,128,16], feature_num=16, n_classes=9,
                  patches=Config(size=(16,16)),
                  transformer=Config(mlp_dim=3072,num_heads=12,num_layers=12,
                                     attention_dropout_rate=0.0,dropout_rate=0.1))
