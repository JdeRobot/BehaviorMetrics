#!/usr/bin/env python3
import npyscreen
import curses

class App(npyscreen.StandardApp):
    def onStart(self):
        self.addForm("MAIN", MainForm, name="Hello Medium!")

class InputBox1(npyscreen.BoxTitle):
    _contained_widget = npyscreen.MultiLineEdit
    def when_value_edited(self):
        self.parent.parentApp.queue_event(npyscreen.Event("event_value_edited"))

class InputBox2(npyscreen.BoxTitle):
    _contained_widget = npyscreen.MultiLineEdit

class MainForm(npyscreen.FormBaseNew):
    def create(self):
        self.add_event_hander("event_value_edited", self.event_value_edited)
        new_handlers = {
            # Set ctrl+Q to exit
            "^Q": self.exit_func,
            # Set alt+enter to clear boxes
            curses.ascii.alt(curses.ascii.NL): self.inputbox_clear
        }
        self.add_handlers(new_handlers)

        y, x = self.useable_space()
        self.InputBox1 = self.add(InputBox1, name="Editable", max_height=y // 2)
        self.InputBox2 = self.add(InputBox2, footer="No editable", editable=False)

    def event_value_edited(self, event):
        self.InputBox2.value = self.InputBox1.value
        self.InputBox2.display()

    def inputbox_clear(self, _input):
        self.InputBox1.value = self.InputBox2.value = ""
        self.InputBox1.display()
        self.InputBox2.display()

    def exit_func(self, _input):
        exit(0)

MyApp = App()
MyApp.run()