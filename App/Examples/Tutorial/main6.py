from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout

class Widgets(Widget):
    pass

class Main6App(App):
    def build(self):
        return Widgets()
    
if __name__ == "__main__":
    Main6App().run()

