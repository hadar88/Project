import json
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.spinner import Spinner
from algorithm import calculate_nutritional_data as cnd

DATA_PATH = "data.json"
d = open(DATA_PATH, "r")
data = json.load(d)
d.close()

#######################################################################

class ColoredLabel(Label):
    def __init__(self, color=(0, 0, 0, 1), text_color=(0, 0, 0, 1), **kw):
        super(ColoredLabel, self).__init__(**kw)
        self.color = text_color  # Set the text color
        with self.canvas.before:
            self.bg_color = Color(*color)  # Use the provided background color
            self.bg_rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_bg, pos=self._update_bg)

    def _update_bg(self, instance, value):
        self.bg_rect.size = self.size
        self.bg_rect.pos = self.pos

#######################################################################

class LoginWindow(Screen):
    def __init__(self, **kw):
        super(LoginWindow, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1)  
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.logo = Image(
            source = "logo.png", size_hint = (0.3, 0.3), pos_hint = {"x": 0.35, "top": 1}
            )
        self.window.add_widget(self.logo)

        self.userName = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "Username", 
            size_hint=(0.8, 0.1), 
            pos_hint={"x": 0.1, "top": 0.68}
        )
        self.window.add_widget(self.userName)

        self.password = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "Password", 
            size_hint=(0.8, 0.1), 
            pos_hint={"x": 0.1, "top": 0.56}, 
            # input_filter="float" 
        )
        self.window.add_widget(self.password)

        self.loginButton = Button(
            text = "Login", 
            font_size = 100, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.44},
            on_press = self.login
        )
        self.window.add_widget(self.loginButton)

        self.errorMassage = ColoredLabel(
            text = "", 
            font_size = 50, 
            size_hint = (0.8, 0.1), 
            pos_hint = {"x": 0.1, "top": 0.32},
            color=(1, 1, 1, 1),
            text_color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.errorMassage)

        self.createAccountButton = Button(
            text = "Create account", 
            font_size = 100, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.2},
            on_press = self.createAccount
        )
        self.window.add_widget(self.createAccountButton)

        ###
        
        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def login(self, instance):
        username = self.userName.text
        password = self.password.text
        self.userName.text = ""
        self.password.text = ""
        if(data["username"] == username and data["password"] == password and username != "" and password != ""):
            self.errorMassage.text = ""
            self.manager.current = "main"
        else:
            self.errorMassage.text = "Invalid username or password"

    def createAccount(self, instance):
        self.errorMassage.text = ""
        self.manager.current = "createAccount"

    def go_home(self, instance):
        self.errorMassage.text = ""
        self.manager.current = "main"

################################

class MainWindow(Screen):
    def __init__(self, **kw):
        super(MainWindow, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1)  
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###



        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

################################

class SecondWindow(Screen):
    def __init__(self, **kw):
        super(SecondWindow, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.home = Button(
            background_normal="home.png",
            size_hint=(0.1125, 0.07), 
            pos_hint={"x": 0, "top": 1},
            on_press=self.go_home
        )
        self.window.add_widget(self.home)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def go_home(self, instance):
        self.manager.current = "main"

################################

class CreateAccountWindow(Screen):
    def __init__(self, **kw):
        super(CreateAccountWindow, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.title = ColoredLabel(
            text = "Create a new account", 
            font_size = 150, 
            size_hint = (0.8, 0.2), 
            pos_hint = {"x": 0.1, "top": 0.9},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.title)

        self.userName = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "Username", 
            size_hint=(0.8, 0.1), 
            pos_hint={"x": 0.1, "top": 0.62}
        )
        self.window.add_widget(self.userName)

        self.password = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "Password", 
            size_hint=(0.8, 0.1), 
            pos_hint={"x": 0.1, "top": 0.5}
        )
        self.window.add_widget(self.password)

        self.login = Button(
            text = "Login", 
            font_size = 90, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.32},
            on_press = self.log_in
        )
        self.window.add_widget(self.login)

        self.submit = Button(
            text = "Submit", 
            font_size = 90, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.8, 0.1),
            pos_hint = {"x": 0.1, "top": 0.2},
            on_press = self.registration
        )
        self.window.add_widget(self.submit)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def log_in(self, instance):
        self.userName.text = ""
        self.password.text = ""
        self.manager.current = "login"

    def registration(self, instance):
        username = self.userName.text
        password = self.password.text
        self.userName.text = ""
        self.password.text = ""
        data["username"] = username
        data["password"] = password
        with open(DATA_PATH, "w") as file:
            json.dump(data, file)
        self.manager.current = "registration1"

################################

class Registration1Window(Screen):
    def __init__(self, **kw):
        super(Registration1Window, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.title = ColoredLabel(
            text = "Registration", 
            font_size = 150, 
            size_hint = (0.8, 0.2), 
            pos_hint = {"x": 0.1, "top": 0.9},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.title)

        self.weightLabel = ColoredLabel(
            text = "Weight:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.6},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.weightLabel)

        self.weightInput = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "Kg", 
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.6}
        )
        self.window.add_widget(self.weightInput)

        self.heightLabel = ColoredLabel(
            text = "Height:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.48},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.heightLabel)

        self.heightInput = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "cm", 
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.48}
        )
        self.window.add_widget(self.heightInput)

        self.genderLabel = ColoredLabel(
            text = "Gender:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.36},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.genderLabel)

        self.genderInput = Spinner(
            text="Select a gender",
            values=("Male", "Female"),
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.36},
            font_size = 50  

        )
        self.window.add_widget(self.genderInput)

        self.nextPage = Button(
            text = "Next page", 
            font_size = 50, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.2},
            on_press = self.next
        )
        self.window.add_widget(self.nextPage)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def next(self, instance):
        weight_input = self.weightInput.text
        height_input = self.heightInput.text
        gender_input = self.genderInput.text
        data["weight"] = weight_input
        data["height"] = height_input
        data["gender"] = gender_input
        with open(DATA_PATH, "w") as file:
            json.dump(data, file)
        self.manager.current = "registration2"

################################

class Registration2Window(Screen):
    def __init__(self, **kw):
        super(Registration2Window, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###



        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

################################

class Registration3Window(Screen):
    def __init__(self, **kw):
        super(Registration3Window, self).__init__(**kw)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1) 
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

################################

class WindowManager(ScreenManager):
    def __init__(self, **kw):
        super(WindowManager, self).__init__(**kw)

        self.add_widget(LoginWindow(name = "login"))
        self.add_widget(MainWindow(name = "main"))
        self.add_widget(SecondWindow(name = "second"))
        self.add_widget(CreateAccountWindow(name = "createAccount"))
        self.add_widget(Registration1Window(name = "registration1"))
        self.add_widget(Registration2Window(name = "registration2"))
        self.add_widget(Registration3Window(name = "registration3"))
        
class MainApp(App):
    def build(self):
        return WindowManager()

if __name__ == "__main__":
    MainApp().run()
