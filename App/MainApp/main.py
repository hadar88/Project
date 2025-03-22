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
from kivy.uix.checkbox import CheckBox

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
            self.manager.current = data["stage"]
        else:
            self.errorMassage.text = "Invalid username or password"

    def createAccount(self, instance):
        self.errorMassage.text = ""
        data["stage"] = "createAccount"
        with open(DATA_PATH, "w") as file:
            json.dump(data, file)
        self.manager.current = "createAccount"
        
    def go_home(self, instance):
        self.errorMassage.text = ""
        data["stage"] = "main"
        with open(DATA_PATH, "w") as file:
            json.dump(data, file)
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

        self.temp = ColoredLabel(
            text = "Main", 
            font_size = 50, 
            size_hint = (0.4, 0.4), 
            pos_hint = {"x": 0.3, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.temp)

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

        self.temp = ColoredLabel(
            text = "Second", 
            font_size = 50, 
            size_hint = (0.4, 0.4), 
            pos_hint = {"x": 0.3, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.temp)

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
            text = "Create an account", 
            font_size = 100, 
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
            pos_hint={"x": 0.1, "top": 0.68}
        )
        self.window.add_widget(self.userName)

        self.password = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "Password", 
            size_hint=(0.8, 0.1), 
            pos_hint={"x": 0.1, "top": 0.56}
        )
        self.window.add_widget(self.password)

        self.errorMassage = ColoredLabel(
            text = "", 
            font_size = 50, 
            size_hint = (0.8, 0.1), 
            pos_hint = {"x": 0.1, "top": 0.44},
            color=(1, 1, 1, 1),
            text_color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.errorMassage)

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
        data["stage"] = "login"
        with open(DATA_PATH, "w") as file:
            json.dump(data, file)
        self.manager.current = "login"
        self.errorMassage.text = ""

    def registration(self, instance):
        username = self.userName.text
        password = self.password.text
        if(username == "" or password == ""):
            self.errorMassage.text = "Cannot leave fields empty"
        else:
            self.userName.text = ""
            self.password.text = ""
            data["username"] = username
            data["password"] = password
            data["stage"] = "registration1"
            with open(DATA_PATH, "w") as file:
                json.dump(data, file)
            self.manager.current = "registration1"
            self.errorMassage.text = ""

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
            size_hint = (0.775, 0.2), 
            pos_hint = {"x": 0.1125, "top": 0.95},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.title)

        self.weightLabel = ColoredLabel(
            text = "Weight:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.weightLabel)

        self.weightInput = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "Kg", 
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.7},
            input_filter="float"
        )
        self.window.add_widget(self.weightInput)

        self.heightLabel = ColoredLabel(
            text = "Height:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.58},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.heightLabel)

        self.heightInput = TextInput(
            multiline = False, 
            font_size = 50, 
            hint_text = "cm", 
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.58},
            input_filter="float"
        )
        self.window.add_widget(self.heightInput)

        self.genderLabel = ColoredLabel(
            text = "Gender:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.46},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.genderLabel)

        self.genderInput = Spinner(
            text="Select a gender",
            values=("Male", "Female"),
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.46},
            font_size = 50  

        )
        self.window.add_widget(self.genderInput)

        self.errorMessage = ColoredLabel(
            text = "", 
            font_size = 50, 
            size_hint = (0.8, 0.1), 
            pos_hint = {"x": 0.1, "top": 0.3},
            color=(1, 1, 1, 1),
            text_color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.errorMessage)

        self.nextPage = Button(
            text = "Next page", 
            font_size = 50, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.14},
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
        if(weight_input == "" or height_input == "" or gender_input == "Select a gender"):
            self.errorMessage.text = "Please fill in all fields"
        else:
            data["weight"] = weight_input
            data["height"] = height_input
            data["gender"] = gender_input
            data["stage"] = "registration2"
            with open(DATA_PATH, "w") as file:
                json.dump(data, file)
            self.manager.current = "registration2"
            self.errorMessage.text = ""

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

        self.home = Button(
            background_normal="back.png",
            size_hint=(0.1125, 0.07), 
            pos_hint={"x": 0, "top": 1},
            on_press=self.previous
        )
        self.window.add_widget(self.home)

        self.title = ColoredLabel(
            text = "Registration", 
            font_size = 150, 
            size_hint = (0.775, 0.2), 
            pos_hint = {"x": 0.1125, "top": 0.95},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.title)

        self.activityLabel = ColoredLabel(
            text = "I am:", 
            font_size = 60, 
            size_hint = (0.44, 0.1), 
            pos_hint = {"x": 0.05, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.activityLabel)

        self.activityInput = Spinner(
            text="Level of activity",
            values=("sedentary", 
                    "lightly active", 
                    "moderately active", 
                    "active", 
                    "extremely active"),
            size_hint=(0.44, 0.1), 
            pos_hint={"x": 0.51, "top": 0.7},
            font_size = 40  
        )
        self.window.add_widget(self.activityInput)

        self.activityTypeLabel = ColoredLabel(
            text = "Types of activity:", 
            font_size = 60, 
            size_hint = (0.6, 0.1), 
            pos_hint = {"x": 0.2, "top": 0.58},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.activityTypeLabel)

        self.cardioLabel = ColoredLabel(
            text = "Cardio", 
            font_size = 50, 
            size_hint = (0.2, 0.05), 
            pos_hint = {"x": 0.3, "top": 0.46},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.cardioLabel)

        self.cardioInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.485},
            color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.cardioInput)

        self.strengthLabel = ColoredLabel(
            text = "Strength", 
            font_size = 50, 
            size_hint = (0.2, 0.05), 
            pos_hint = {"x": 0.3, "top": 0.39},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.strengthLabel)

        self.strengthInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.415},
            color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.strengthInput)

        self.muscleLabel = ColoredLabel(
            text = "Muscle", 
            font_size = 50, 
            size_hint = (0.2, 0.05), 
            pos_hint = {"x": 0.3, "top": 0.32},
            color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.muscleLabel)

        self.muscleInput = CheckBox(
            size_hint=(0.1, 0.1),
            pos_hint={"x": 0.55, "top": 0.345},
            color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.muscleInput)

        self.errorMessage = ColoredLabel(
            text = "", 
            font_size = 50, 
            size_hint = (0.8, 0.1), 
            pos_hint = {"x": 0.1, "top": 0.255},
            color=(1, 1, 1, 1),
            text_color=(1, 0, 0, 1)
        )
        self.window.add_widget(self.errorMessage)

        self.nextPage = Button(
            text = "Next page", 
            font_size = 50, 
            background_color = (1, 1, 1, 1), 
            # background_normal = "",
            size_hint = (0.4, 0.1),
            pos_hint = {"x": 0.3, "top": 0.14},
            on_press = self.next
        )
        self.window.add_widget(self.nextPage)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def next(self, instance):
        activity = self.activityInput.text
        cardio = self.cardioInput.active
        strength = self.strengthInput.active
        muscle = self.muscleInput.active
        if(activity == "Level of activity"):
            self.errorMessage.text = "Please select a level of activity"
        else:
            data["activity"] = activity
            data["cardio"] = cardio
            data["strength"] = strength
            data["muscle"] = muscle
            data["stage"] = "registration3"
            with open(DATA_PATH, "w") as file:
                json.dump(data, file)
            self.manager.current = "registration3"
            self.errorMessage.text = ""

    def previous(self, instance):
        self.manager.current = "registration1"

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

        self.temp = ColoredLabel(
            text = "Registration3", 
            font_size = 50, 
            size_hint = (0.4, 0.4), 
            pos_hint = {"x": 0.3, "top": 0.7},
            color=(0, 0, 1, 1),
            text_color=(0, 0, 0, 1)
        )
        self.window.add_widget(self.temp)

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
