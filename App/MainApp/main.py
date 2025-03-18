from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Color, Rectangle
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.spinner import Spinner

class LoginWindow(Screen):
    def __init__(self, **kw):
        super(LoginWindow, self).__init__(**kw)

        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1)  # Set the color to white
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.logo = Image(
            source = "logo.png", size_hint = (0.25, 0.25), pos_hint = {"x": 0.375, "top": 0.98}
            )
        self.window.add_widget(self.logo)

        self.userName = TextInput(
            multiline = False, font_size = 14, hint_text = "Username", size_hint=(0.8, 0.1), pos_hint={"x": 0.1, "top": 0.71}
        )
        self.window.add_widget(self.userName)

        self.password = TextInput(
            multiline = False, font_size = 14, hint_text = "Password", size_hint=(0.8, 0.1), pos_hint={"x": 0.1, "top": 0.59}, input_filter="float", keyboard_mode='auto'
        )
        self.window.add_widget(self.password)

        self.loginButton = Button(
            text = "Login", 
            font_size = 14, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.47},
            # on_press = self.login
        )
        self.window.add_widget(self.loginButton)

        # self.gender_spinner = Spinner(
        #     text="Select Gender",
        #     values=("Male", "Female", "Other"),
        #     size_hint=(0.8, 0.1),
        #     pos_hint={"x": 0.1, "top": 0.32},
        # )
        # self.window.add_widget(self.gender_spinner)

        self.createAccountButton = Button(
            text = "Create account", 
            font_size = 14, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.2},
            # on_press = self.login
        )
        self.window.add_widget(self.createAccountButton)

        ###
        
        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size


class WindowManager(ScreenManager):
    def __init__(self, **kwargs):
        super(WindowManager, self).__init__(**kwargs)

        self.add_widget(LoginWindow(name = "login"))
        
class MainApp(App):
    def build(self):
        return WindowManager()

if __name__ == "__main__":
    MainApp().run()
