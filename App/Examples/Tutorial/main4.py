from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Color, Line, Triangle, Ellipse

class Touch(Widget):
    def __init__(self, **kwargs):
        super(Touch, self).__init__(**kwargs)

        with self.canvas:
            Color(0, 1, 0, 0.5, mode = 'rgba')
            self.line = Line(points = (50, 50, 300, 300))
            Color(1, 0, 0, 0.5, mode = 'rgba')
            self.rect1 = Rectangle(pos = (0, 0), size = (50, 50))
            self.rect2 = Rectangle(pos = (300, 300), size_hint = (0.05, 0.05))
            self.tria1 = Triangle(points = (100, 100, 150, 150, 200, 100))
            self.tria2 = Triangle(points = (200, 200, 250, 250, 300, 200))  
            self.elli1 = Ellipse(pos = (400, 400), size = (100, 50))

    def on_touch_down(self, touch):
        self.rect1.pos = touch.pos
        # print("Mouse Down", touch)
    
    def on_touch_move(self, touch):
        self.rect1.pos = touch.pos
        # print("Mouse Move", touch)

class Main4App(App):
    def build(self):
        return Touch()
    

if __name__ == "__main__":
    Main4App().run()