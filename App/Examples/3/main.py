import kivy
from kivy.app import App
from  kivy.uix.button import Button
from kivy.uix.widget import Widget

# 2:
# class FunckyButton(Button):
#     def __init__(self, **kwargs):
#         super(FunckyButton, self).__init__(**kwargs)
#         self.text = "Funky Button"
#         self.pos = (100, 100)
#         self.size_hint = (.25, .25)
# 3:
# class FunkyButton(Button):
#     pass
# 4, 5:
class GameScreen(Widget):
    pass

class MainApp(App):
    def build(self):
        # 1:
        # return Button(        
        #     text="Hello World",
        #     pos = (50, 50), # where it start: (0, 0) is the left bottom corner
        #     size = (500, 500), # static size 
        #     size_hint = (None, None) # what percentage of the parent's width and height the widget is going to take, can use (None, None) for manual size
        #     )
        # 2: 
        # return FunckyButton()
        # 3:
        # return FunkyButton(
        #     pos = (100, 100),
        #     size_hint = (None, None),
        #     size = (500, 500)
        # )
        # 4, 5:
        return GameScreen()
        
    

if __name__ == "__main__":
    MainApp().run()
