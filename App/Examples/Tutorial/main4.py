from kivy.app import App
from kivy.uix.widget import Widget

class Touch(Widget):

    def on_touch_down(self, touch):
        print("Mouse Down", touch)
    
    def on_touch_move(self, touch):
        print("Mouse Move", touch)

    def on_touch_up(self, touch):
        print("Mouse Up", touch)

class Main4App(App):
    def build(self):
        return Touch()
    

if __name__ == "__main__":
    Main4App().run()