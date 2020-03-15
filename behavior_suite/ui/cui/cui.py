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

class CUI:

    last_key = None

    def __init__(self):
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        # self.comm = Communicator()

    def on_press(self, key):
        try:
            # print('Pressed key'.format(key.char))
            # self.comm.send_msg(key.char)
            self.last_key = key.char
        except AttributeError:
            # print('Not alphanumeric'.format(key))
            pass

    def on_release(self, key):
        # print('Released'.format(key))
        if key == keyboard.Key.esc:
            self.last_key = '-1'
            # self.comm.send_msg('quit')
            return False

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()

    def set_dset_dir(self, dset_out):
        pass

    def set_dset_name(self, dset_name):
        pass
    def set_topics(self, topics_box):
        pass
    
    def set_brain(self, brain):
        pass