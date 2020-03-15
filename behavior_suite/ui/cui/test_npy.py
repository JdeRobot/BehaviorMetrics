import npyscreen
import threading
import time

from cui import CUI

# This application class serves as a wrapper for the initialization of curses
# and also manages the actual forms of the application

cui = CUI()
cui.start()
brains_dir = '/home/fran/github/BehaviorSuite/behavior_suite/brains/f1/brain_f1_opencv.py'
dset_dir = '/home/fran/github/BehaviorSuite/behavior_suite/datasets/'
logs_path = '/home/fran/github/BehaviorSuite/behavior_suite/logs/log.txt'
ros_topics = ['/F1ROS/cmd_vel', '/F1ROS/cameraL/image_raw', '/F1ROS/odom', '/F1ROS/laser/scan']
cont = 0

class MyTestApp(npyscreen.StandardApp):
    def onStart(self):
        self.form = self.addForm("MAIN", MainForm)

    def stop(self):
        self.queue_event(npyscreen.Event("exit_app"))

class InputBox2(npyscreen.BoxTitle):
    _contained_widget = npyscreen.BufferPager

class MainForm(npyscreen.FormMultiPage):

    commands = ["p","r","d","s","c","e"]
    kill_event = threading.Event()
    log_data = ""
    logs = open(logs_path, 'r')
    data = []
    
    def event_value_edited(self, event):
        self.status.value = 'Last command: ' + cui.last_key
        self.status.display()
    
    def log_update(self, event):
        self.data.append(self.logs.readline())
        self.log_box.buffer(self.data)
        self.log_box.display()
        # prev = self.log_data
        # self.log_data = self.logs.read()
        # self.log_box.value = self.log_data
        # self.log_box.display()
        pass
    
    def clear_logs(self,event):
        self.data = []
    
    def exit_func(self, _input):
        self.logs.close()
        self.kill_event.set()
        exit(0)

    def create(self):
        self.add_event_hander("event_value_edited", self.event_value_edited)
        self.add_event_hander("exit_app", self.exit_func)
        self.add_event_hander("log_update", self.log_update)
        new_handlers = {
            # Set ctrl+Q to exit
            '^Q': self.exit_func,
            '^L': self.clear_logs
        }
        self.add_handlers(new_handlers)

        self.name = "Welcome to Behavior Suite"
        self.FIX_MINIMUM_SIZE_WHEN_CREATED = True
        commands_box = self.add(
            npyscreen.BoxTitle,
            max_width=35,
            max_height=14,
            name="Keyboard commands:",
            scroll_exit = True,
            editable=False,
            color='VERYGOOD',
            contained_widget_arguments = {
                'color': "LABEL", 
                'widgets_inherit_color': True,}
            )
        commands_box.values = [
            "",
            "p - Pause simulation", 
            "r - Resume simulation", 
            "d - Start recording dataset", 
            "s - Stop recording dataset",
            "c - Change brain",
            "e - Evaluate brain",
            "Ctrl+L - Clear logs",
            "----------------------------------",
            "Ctrl+Q - Exit program"
        ]
        logs_title = self.add(npyscreen.FixedText, value='Logs', editable=False, color='CAUTIONHL')
        self.log_box = self.add(
            npyscreen.BufferPager,
            name="Logs",
            footer="Logs saved in: " + logs_path,
            editable=False,
            max_height=15,
            color='CAUTIONHL',
            values='Log',
            scroll_end=True,
            scroll_if_editing=False,
            # rely=14
            relx=4
        )
        self.parameters_title = self.add(npyscreen.FixedText,
            value='Parameters',
            rely=2,
            relx=38,
            scroll_exit = True,
            editable=False,
            color='CRITICAL'
        )        
        self.brain = self.add(npyscreen.TitleFilenameCombo,
            name="Brain:",
            rely=4,
            relx=38,
            select_dir=False,
            must_exist=False,
            confirm_if_exists=True,
            sort_by_extension=True,
            value= brains_dir,
            color='CURSOR'
        )
        self.dset_out = self.add(npyscreen.TitleFilenameCombo,
            name="Dataset out:",
            # rely=6,
            relx=38,
            select_dir=True,
            must_exist=False,
            confirm_if_exists=True,
            sort_by_extension=True,
            value= dset_dir
        )
        self.dset_name = self.add(npyscreen.TitleFilename,
            name = "Dataset name:",
            relx=38,
            value= 'default.bag',
            color='CURSOR'
        )
        self.topics_box = self.add (npyscreen.TitleMultiSelect,
            relx=38,
            max_height =4,
            value = [1,],
            name="ROS Topics",
            values = ros_topics,
            scroll_exit=True,
        
        )
        self.info = self.add(npyscreen.FixedText,
            value='For dataset recording (press space to select).',
            # rely=29,
            relx=38,
            scroll_exit = True,
            editable=False,
            color='LABEL'
        )

        self.status = self.add(npyscreen.FixedText,
            value='Last command: None',
            # rely=29,
            relx=38,
            scroll_exit = True,
            editable=False,
            color='CONTROL'
        )

        self.t_update = threading.Thread(target=self.update_form)
        self.t_update.start()

        self.edit()

    def update_form(self):
        global cont 
        while not self.kill_event.isSet():
            
            cont += 1
            self.update_info()
            self.parentApp.queue_event(npyscreen.Event("log_update"))
            if cui.last_key in self.commands:
                self.parentApp.queue_event(npyscreen.Event("event_value_edited"))
            elif cui.last_key == '-1':
                break
            time.sleep(1)
    
    def update_info(self):
        global cui
        cui.set_dset_dir(self.dset_out.value)
        cui.set_dset_name(self.dset_name.value)
        cui.set_topics([ros_topics[i] for i in self.topics_box.value])
        cui.set_brain(self.brain.value)
        
if __name__ == '__main__':
    try:
        TA = MyTestApp()
        TA.run()
    except KeyboardInterrupt:
        print("Exiting... Press Esc to confirm")
        TA.stop()
        exit(0)