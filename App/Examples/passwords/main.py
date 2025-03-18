import json
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics import Color, Rectangle
from kivy.uix.floatlayout import FloatLayout

DATA_PATH = "data.json"
d = open(DATA_PATH, "r")
data = json.load(d)
d.close()

class ColoredLabel(Label):
    def __init__(self, color=(0, 0, 0, 1), **kwargs):
        super(ColoredLabel, self).__init__(**kwargs)
        with self.canvas.before:
            self.bg_color = Color(*color)  # Use the provided color
            self.bg_rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_bg, pos=self._update_bg)

    def _update_bg(self, instance, value):
        self.bg_rect.size = self.size
        self.bg_rect.pos = self.pos

class MainWindow(Screen):
    def __init__(self, **kw):
        super(MainWindow, self).__init__(**kw)
        self.cols = 1

        self.window = GridLayout(cols = 1, size_hint = (1, 1))

        # the color is : Color(0.047, 0.035, 0.87, 1)
        self.label1 = ColoredLabel(
            text = "Password Manager", font_size = 100, color = (0.047, 0.035, 0.87, 1)
            )
        self.window.add_widget(self.label1)
        
        self.button1 = Button(
            text = "Search for password", 
            font_size = 50, 
            background_color = (0, 1, 0, 1), 
            size_hint_y = 0.3, 
            on_press = self.search
            )
        self.window.add_widget(self.button1)

        self.button2 = Button(
            text = "Add password", 
            font_size = 50, 
            background_color = (0.0353, 0.8, 0.87, 1), 
            background_normal = "", 
            size_hint_y = 0.3, 
            on_press = self.add
            )
        self.window.add_widget(self.button2)

        self.add_widget(self.window)

    def search(self, instance):
        # move to the search window
        self.manager.current = "search"
    
    def add(self, instance):
        # move to the add window
        self.manager.current = "add"


class SearchWindow(Screen):
    def __init__(self, **kw):
        super(SearchWindow, self).__init__(**kw)

        self.window = FloatLayout(size_hint=(1, 1))

        self.label1 = ColoredLabel(
            text = "Search a password", font_size = 100, size_hint=(1, 0.3), pos_hint={"x": 0, "top": 1}, color = (1, 0, 0, 1)
        )
        self.window.add_widget(self.label1)

        self.name_input = TextInput(
            multiline = False, font_size = 60, hint_text = "Name", size_hint=(1, 0.2), pos_hint={"x": 0, "top": 0.7}
            )
        self.window.add_widget(self.name_input)

        self.label2 = ColoredLabel(
            text = "Password:", font_size = 40, size_hint=(0.5, 0.2), pos_hint={"x": 0, "top": 0.5}, color = (0.047, 0.035, 0.87, 1)
        )
        self.window.add_widget(self.label2)

        self.label2 = ColoredLabel(
            text = "", font_size = 40, size_hint=(0.5, 0.2), pos_hint={"x": 0.5, "top": 0.5}, color = (0.8588, 0.478, 0.043, 1)
        )
        self.window.add_widget(self.label2) 

        self.search_button = Button(
            text = "Search", 
            font_size = 50, 
            background_color = (0.0353, 0.8, 0.87, 1), 
            background_normal = "", 
            size_hint=(1, 0.3),
            pos_hint={"x": 0, "top": 0.3},
            on_press = self.search
        )
        self.window.add_widget(self.search_button)

        self.return_button = Button(
            text = "<--",
            font_size = 40,
            background_color = (0.047, 0.035, 0.87, 1),
            background_normal = "",
            size_hint = (None, None),  
            size = (100, 100),       
            pos_hint = {"x": 0, "top": 1},
            on_press = self.return_to_main
        )
        self.window.add_widget(self.return_button)

        self.add_widget(self.window)

    def return_to_main(self, instance):
        self.label2.text = ""
        self.manager.current = "main"

    def search(self, instance):
        name = self.name_input.text
        self.name_input.text = ""
        try:
            password = data[name]
            self.label2.text = password
        except:
            self.label2.text = "Password not found"

class AddWindow(Screen):
    def __init__(self, **kw):
        super(AddWindow, self).__init__(**kw)
        
        self.cols = 1   

        self.window = FloatLayout(size_hint=(1, 1))

        self.label = ColoredLabel(
            text = "Add a password", font_size = 100, size_hint=(1, 0.3), pos_hint={"x": 0, "top": 1}, color = (1, 0, 0, 1)
            )
        self.window.add_widget(self.label)

        self.name_input = TextInput(
            multiline = False, font_size = 60, hint_text = "Name", size_hint=(1, 0.2), pos_hint={"x": 0, "top": 0.7}
            )
        self.window.add_widget(self.name_input)

        self.passw = TextInput(
            multiline = False, font_size = 60, hint_text = "Password", size_hint=(1, 0.2), pos_hint={"x": 0, "top": 0.5}
            )
        self.window.add_widget(self.passw)

        self.button = Button(
            text = "Add", 
            font_size = 50, 
            background_color = (0.0353, 0.8, 0.87, 1), 
            background_normal = "", 
            size_hint=(1, 0.3),
            pos_hint={"x": 0, "top": 0.3},
            on_press = self.add
            )
        self.window.add_widget(self.button)

        self.return_button = Button(
            text = "<--",
            font_size = 40,
            background_color = (0.047, 0.035, 0.87, 1),
            background_normal = "",
            size_hint = (None, None),  
            size = (100, 100),       
            pos_hint = {"x": 0, "top": 1},
            on_press = self.return_to_main
        )
        self.window.add_widget(self.return_button)

        self.add_widget(self.window)

    def add(self, instance):
        name = self.name_input.text
        password = self.passw.text

        data[name] = password

        with open(DATA_PATH, "w") as d:
            json.dump(data, d)

        self.name_input.text = ""
        self.passw.text = ""

    def return_to_main(self, instance):
        self.manager.current = "main"


class WindowManager(ScreenManager):
    def __init__(self, **kwargs):
        super(WindowManager, self).__init__(**kwargs)

        self.add_widget(MainWindow(name = "main"))
        self.add_widget(SearchWindow(name = "search"))
        self.add_widget(AddWindow(name = "add"))     

class MainApp(App):
    def build(self):
        return WindowManager()
    
if __name__ == "__main__":
    MainApp().run()