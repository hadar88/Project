from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Color, Rectangle
from kivy.uix.textinput import TextInput
from kivy.uix.video import Video


class Grid1(GridLayout):
    def __init__(self, screen_manager, **kwargs):
        super(Grid1, self).__init__(**kwargs)
        self.cols = 1

        with self.canvas.before:
            Color(0, 0.55, 1, 1) 
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)
        
        self.label = Label(text="Enter credit card data:",
                            font_size= 80,
                            color = (0, 0, 0, 1)
                            )
        self.add_widget(self.label)

        self.add_widget(Grid2())
        self.add_widget(Grid3(screen_manager))
        
    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

class Grid2(GridLayout):
    def __init__(self, **kwargs):
        super(Grid2, self).__init__(**kwargs)
        self.cols = 2

        with self.canvas.before:
            Color(0.7, 0, 0, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)

        self.label1 = Label(text="Card number:",
                            font_size= 40,
                            color = (0, 0, 0, 1)
                            )
        self.input1 = TextInput(multiline = False)
        self.add_widget(self.label1)
        self.add_widget(self.input1)

        self.label2 = Label(text="Expiration date:",
                            font_size= 40,
                            color = (0, 0, 0, 1)
                            )
        self.input2 = TextInput(multiline = False)
        self.add_widget(self.label2)
        self.add_widget(self.input2)

        self.label3 = Label(text="CVV:",
                            font_size= 40,
                            color = (0, 0, 0, 1)
                            )
        self.input3 = TextInput(multiline = False)
        self.add_widget(self.label3)
        self.add_widget(self.input3)
        
    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

class Grid3(GridLayout):
    def __init__(self, screen_manager, **kwargs):
        super(Grid3, self).__init__(**kwargs)
        self.cols = 1
        self.screen_manager = screen_manager

        # with self.canvas.before:
        #     Color(0, 1, 0, 1)
        #     self.rect = Rectangle(size=self.size, pos=self.pos)
        # self.bind(size=self._update_rect, pos=self._update_rect)

        self.button = Button(text="Submit",
                             font_size= 80,
                             color = (0, 0, 0, 1),
                             background_color = (0, 1, 0, 1)
                             )
        self.button.bind(on_press = self.pressed)
        self.add_widget(self.button)

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

    def pressed(self, instance):
        # it should move to the second window
        self.screen_manager.current = "second"

        
class MainWindow(Screen):
    def __init__(self, screen_manager, **kw):
        super(MainWindow, self).__init__(**kw)
        self.name = "main"
        self.add_widget(Grid1(screen_manager))


class SecondWindow(Screen):
    def __init__(self, **kw):
        super(SecondWindow, self).__init__(**kw)
        self.name = "second"
        # put the song in the second window
        self.song = Video(source = "song.mp4", state = "stop")
        self.add_widget(self.song)
    
    def on_enter(self, *args):
        # Start the video when entering the screen
        self.song.state = "play"
        super(SecondWindow, self).on_enter(*args)


class WindowManager(ScreenManager):
    def __init__(self, **kwargs):
        super(WindowManager, self).__init__(**kwargs)
        self.main_window = MainWindow(self)
        self.second_window = SecondWindow()
        self.add_widget(self.main_window)
        self.add_widget(self.second_window)
        

class MainApp(App):
    def build(self):
        return WindowManager()

if __name__ == "__main__":
    MainApp().run()
