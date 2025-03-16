import json
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics import Color, Rectangle

DATA_PATH = "data.json"

class ColoredLabel1(Label):
    def __init__(self, **kwargs):
        super(ColoredLabel1, self).__init__(**kwargs)
        with self.canvas.before:
            self.bg_color = Color(0.047, 0.035, 0.87, 1) 
            self.bg_rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_bg, pos=self._update_bg)

    def _update_bg(self, instance, value):
        self.bg_rect.size = self.size
        self.bg_rect.pos = self.pos

class ColoredLabel2(Label):
    def __init__(self, **kwargs):
        super(ColoredLabel2, self).__init__(**kwargs)
        with self.canvas.before:
            self.bg_color = Color(1, 0, 0, 1) 
            self.bg_rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_bg, pos=self._update_bg)

    def _update_bg(self, instance, value):
        self.bg_rect.size = self.size
        self.bg_rect.pos = self.pos

class LoginWindow(Screen):
    def __init__(self, **kw):
        super(LoginWindow, self).__init__(**kw)
        self.cols = 1

        self.window = GridLayout(cols = 1, size_hint = (1, 1))

        # the label height should be 80% of the window height
        self.label1 = ColoredLabel1(text = "Password Manager", font_size = 40)
        self.window.add_widget(self.label1)
        
        self.button1 = Button(
            text = "Search for password", 
            font_size = 40, 
            background_color = (0, 1, 0, 1), 
            size_hint_y = 0.3, 
            on_press = self.search
            )
        self.window.add_widget(self.button1)

        self.button2 = Button(
            text = "Add password", 
            font_size = 40, 
            background_color = (0.0353, 0.8, 0.87, 1), 
            background_normal = "", 
            size_hint_y = 0.3, 
            on_press = self.add
            )
        self.window.add_widget(self.button2)

        self.add_widget(self.window)

    def search(self, instance):
        # move to the search window
        pass
    
    def add(self, instance):
        # move to the add window
        self.manager.current = "add"


class SearchWindow(Screen):
    def __init__(self, **kw):
        super(SearchWindow, self).__init__(**kw)

class AddWindow(Screen):
    def __init__(self, **kw):
        super(AddWindow, self).__init__(**kw)

        d = open(DATA_PATH, "r")
        self.data = json.load(d)
        d.close()

        self.cols = 1   

        self.window = GridLayout(cols = 1, size_hint = (1, 1))

        self.label = ColoredLabel2(text = "Add a password", font_size = 40)
        self.window.add_widget(self.label)

        self.name_input = TextInput(multiline = False, font_size = 40, hint_text = "Name")
        self.window.add_widget(self.name_input)

        self.passw = TextInput(multiline = False, font_size = 40, hint_text = "Password")
        self.window.add_widget(self.passw)

        self.button = Button(
            text = "Add", 
            font_size = 40, 
            background_color = (0.0353, 0.8, 0.87, 1), 
            background_normal = "", 
            on_press = self.add
            )
        self.window.add_widget(self.button)

        self.add_widget(self.window)

    def add(self, instance):
        name = self.name_input.text
        password = self.passw.text

        self.data[name] = password

        with open(DATA_PATH, "w") as d:
            json.dump(self.data, d)

        self.name_input.text = ""
        self.passw.text = ""


class WindowManager(ScreenManager):
    def __init__(self, **kwargs):
        super(WindowManager, self).__init__(**kwargs)

        self.add_widget(LoginWindow(name = "login"))
        self.add_widget(SearchWindow(name = "search"))
        self.add_widget(AddWindow(name = "add"))     

class MainApp(App):
    def build(self):
        return WindowManager()
    
if __name__ == "__main__":
    MainApp().run()