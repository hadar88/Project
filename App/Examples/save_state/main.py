from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.utils import get_color_from_hex
from kivy.config import ConfigParser

class Menu(App):
    def build(self):
        self.config = ConfigParser()
        self.config.read('config.ini')

        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (1, 0.7)
        self.window.pos_hint = {"center_x": 0.5, "center_y":0.5}
        Window.clearcolor = get_color_from_hex('#34baeb')

        self.image = Image(
            source=self.config.get('settings', 'image', fallback = "burger.jpeg"),
            size_hint=(0.5, 0.7),
            pos_hint={"center_x": 1, "center_y": 0.5},
            )

        self.window.add_widget(self.image)
        
        self.ask = Label(
            text = "What would you like to eat?",
            font_size = 18,
            color = '#000105'
        )
        self.window.add_widget(self.ask)

        self.answer = TextInput(
            multiline = False,
            padding = (10, 10, 10, 10),
            size_hint = (0.8, 0.3)
        )
        self.window.add_widget(self.answer)

        self.button = Button(
            text = "ORDER",
            size_hint = (1, 0.5),
            bold = True,
            background_color = '#000000'
        )

        self.window.add_widget(self.button)
        self.button.bind(on_press=self.callback)
        return self.window
    
    def callback(self, instance):
        if self.answer.text == "pizza":
            self.image.source = "pizza.jpeg"
            self.saveState()
        elif self.answer.text == "burger":
            self.image.source = "burger.jpeg"
            self.saveState()
        else:
            self.ask.text = "We don't have that on the menu."

    def saveState(self):
        if not self.config.has_section('settings'):
            self.config.add_section('settings') 

        self.config.set('settings', 'image', self.image.source)
        self.config.write()



    def on_stop(self):
        self.saveState()

if __name__ == "__main__":
    Menu().run()