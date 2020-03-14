import npyscreen

# This application class serves as a wrapper for the initialization of curses
# and also manages the actual forms of the application

class MyTestApp(npyscreen.NPSAppManaged):
    def onStart(self):
        self.registerForm("MAIN", MainForm())

# This form class defines the display that will be presented to the user.

class MainForm(npyscreen.Form):
    def create(self):
        F = npyscreen.Form(name = "Welcome to Npyscreen",)
        t = F.add(npyscreen.BoxBasic, name = "Basic Box:", max_width=30, relx=2, max_height=3)
        t.footer = "This is a footer"
        
        t1 = F.add(npyscreen.BoxBasic, name = "Basic Box:", rely=2, relx=32, 
                        max_width=30, max_height=3)
        
        
        t2 = F.add(npyscreen.BoxTitle, name="Box Title:", max_height=6)
        t3 = F.add(npyscreen.BoxTitle, name="Box Title2:", max_height=6,
                        scroll_exit = True,
                        contained_widget_arguments={
                                'color': "WARNING", 
                                'widgets_inherit_color': True,}
                        )
        
        
        t2.entry_widget.scroll_exit = True
        t2.values = ["Hello", 
            "This is a Test", 
            "This is another test", 
            "And here is another line",
            "And here is another line, which is really very long.  abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
            "And one more."]
        t3.values = t2.values

        F.edit()

    def afterEditing(self):
        self.parentApp.setNextForm(None)

if __name__ == '__main__':
    TA = MyTestApp()
    TA.run()