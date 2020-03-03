import sys
import os

import argparse
import yaml

from pilot import Pilot

class Colors:
    """
    Colors defined for improve the prints in each Stage
    """
    DEBUG = '\033[1;36;1m'
    OKCYAN = '\033[96m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def check_args(argv):

    config_data = {}

    parser = argparse.ArgumentParser(description='Neural Behaviors Suite',
                                     epilog='Enjoy the program! :)')

    parser.add_argument('-c',
                        '--config',
                        action='store',
                        type=str,
                        required=True,
                        help=f'{Colors.OKBLUE}Path to the configuration file in YML format.{Colors.ENDC}')

    parser.add_argument('-g',
                        '--gui',
                        action='store_true',
                        help=f'{Colors.OKBLUE}Load the GUI. If not set, the console UI will start by default{Colors.ENDC}')

    parser.add_argument('-l',
                        '--launch',
                        action='store',
                        type=str,
                        required=False,
                        help=f'''{Colors.OKBLUE}Path to the ROS launch file of the desired environment. If not set, the porgram will
                        assume that there is already a simulation or a real robot ready to rock!{Colors.ENDC}''')

    args = parser.parse_args()

    if not os.path.isfile(args.config):
        parser.error(f'{Colors.FAIL}No such file {args.config}{Colors.ENDC}')

    config_data['config'] = args.config

    if args.launch:
        if not os.path.isfile(args.launch):
            parser.error(f'{Colors.FAIL}No such file {args.launch}{Colors.ENDC}')
        config_data['launch'] = args.launch

    return config_data

def launch_env(launch_file):
    pass

def get_config_data(config_file):

    try:
        with open(config_file, 'r') as file:
            cfg = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise SystemExit(f'{Colors.FAIL}Error: Cannot read/parse YML file. Check YAML syntax.{Colors.ENDC}')


if __name__ == '__main__':

    config_data = check_args(sys.argv)

    launch_file = config_data.get('launch', None)
    if launch_file:
        launch_env(launch_file)

    config_file = config_data.get('config', None)
    configuration = get_config_data(config_file)

    pilot = Pilot(configuration)

    '''
    Dos opciones aqui:
        * GUI (Graphic UI) [process]
            - hecho con QtQuick 2.0
            - Inspirado en el de ahora
        * CUI (Console UI) [process]
            - Basado en eventos de teclado
            - Mismas opciones que el GUI

        Ambos envían mensajes al piloto.
    
    GUI (CUI):
        * Grabar dataset (r, record)
        * Cargar dataset (l, load)
        * Pausar piloto (p, pause)
        * Relanzar piloto (u, unpause)
        * Cambiar cerebro (c, change)
        * Evaluar comportamiento (e, evaluate)
    '''
