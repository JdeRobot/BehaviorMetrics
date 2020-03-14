"""
    * Grabar dataset (r, record)
    * Cargar dataset (l, load)
    * Pausar piloto (p, pause) -> abrir el tui
    * Relanzar piloto (u, unpause)
    * Cambiar cerebro (c, change)
    * Evaluar comportamiento (e, evaluate)
"""
from pynput import keyboard

from ui.ros_ui_com import Communicator

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

class CUI:

    def __init__(self):
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.comm = Communicator()
        self.dataset_submenu = False
        self.evaluate_submenu = False

    def on_press(self, key):
        try:
            # print('Pressed key'.format(key.char))
            self.comm.send_msg(key.char)
        except AttributeError:
            # print('Not alphanumeric'.format(key))
            pass

    def on_release(self, key):
        # print('Released'.format(key))
        if key == keyboard.Key.esc:
            self.comm.send_msg('quit')
            return False

    def start(self):
        self.listener.start()


    def show_main_menu():
        pass

