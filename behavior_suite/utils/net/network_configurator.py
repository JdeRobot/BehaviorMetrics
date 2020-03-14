
import importlib

class NetworkConfigurator:

    def __init__(self, cfg_file=None):
        self.cfg = cfg_file
        self.net_options = {
            'framework' : None,
            'type' : None,
            'cropped' : False,
            'model_v_path' : "",
            'model_w_path' : ""
        }
        if self.cfg:
            self.__configure_net()
    
    def config_from_gui(self, net_framework, net_type, net_cropped, net_model_v, net_model_w):

        self.net_options['framework'] = net_framework
        self.net_options['type'] = net_type
        self.net_options['cropped'] = net_cropped
        self.net_options['model_v_path'] = net_model_v
        self.net_options['model_w_path'] = net_model_w

        return self.create_network()


    def __configure_net(self):
        """ fill net configuration dict"""
        net_cfg = self.cfg['Driver']['Network']
        net_framework = net_cfg['Framework']
        net_type = net_cfg['Use']
        net_cropped = net_cfg['Cropped']
        net_model_v = net_cfg['Models_Path'] + '/' + net_cfg['Model_'+net_type+'_v']
        net_model_w = net_cfg['Models_Path'] + '/' + net_cfg['Model_'+net_type+'_w']

        self.net_options['framework'] = net_framework
        self.net_options['type'] = net_type
        self.net_options['cropped'] = net_cropped
        self.net_options['model_v_path'] = net_model_v
        self.net_options['model_w_path'] = net_model_w


    def create_network(self):
        """
        Creates an instance of a default network which config is obtainded from the yml file
        :param cfg: configuration
        :return net: network instance built from configuration options
        :raise SystemExit in case of invalid network
        """

        framework = self.net_options['framework']
        net_type = self.net_options['type']
        net = None

        try:
            module_name = 'net.' + framework.lower() + '.' + net_type.lower() + '.' + net_type.lower() + '_network'
            module_import = importlib.import_module(module_name)
            Net = getattr(module_import, net_type + 'Network')
            net = Net(self.net_options)
        except:
            raise SystemExit('ERROR: Invalid network selected')

        return net