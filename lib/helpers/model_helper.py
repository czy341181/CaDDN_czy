from lib.models.CaDDN import CaDDN

def build_model(cfg):
    if cfg['type'] == 'CaDDN':
        return CaDDN(cfg)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])


