import re
import requests
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import AsyncImage
from kivy.utils import platform
from kivy.clock import mainthread

if platform == "android":
    from jnius import autoclass
    PythonActivity = autoclass("org.kivy.android.PythonActivity")
    Intent = autoclass("android.content.Intent")
    Uri = autoclass("android.net.Uri")

# Google OAuth Client ID (Replace with your actual client ID)
CLIENT_ID = "211621332597-dlidmr7mb13fc20nln7p42lpn05h6srn.apps.googleusercontent.com"
REDIRECT_URI = "com.hadar.NutriPlan:/oauth2redirect"


class LoginScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)

        # Title
        self.label = Label(text="Google Login Example", font_size=24)
        self.add_widget(self.label)

        # Login Button
        self.login_btn = Button(text="Login with Google", size_hint=(1, 0.2))
        self.login_btn.bind(on_press=self.google_login)
        self.add_widget(self.login_btn)

        # Profile Image
        self.profile_image = AsyncImage(size_hint=(1, 0.4))
        self.add_widget(self.profile_image)

        # User Info Label
        self.user_info_label = Label(text="", font_size=18)
        self.add_widget(self.user_info_label)

        # Register intent listener for handling OAuth redirect
        if platform == "android":
            self.register_intent_listener()

    def google_login(self, instance):
        """ Open Google OAuth login inside WebView """
        auth_url = (
            f"https://accounts.google.com/o/oauth2/auth?"
            f"response_type=token&client_id={CLIENT_ID}"
            f"&redirect_uri={REDIRECT_URI}"
            f"&scope=email%20profile"
        )

        if platform == "android":
            self.open_url_android(auth_url)
        else:
            import webbrowser
            webbrowser.open(auth_url)

    def open_url_android(self, url):
        """ Open URL in WebView or default browser on Android """
        intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
        PythonActivity.mActivity.startActivity(intent)

    def register_intent_listener(self):
        """ Register an intent listener to capture OAuth redirects """
        activity = PythonActivity.mActivity
        activity.bind(on_new_intent=self.on_new_intent)

    def on_new_intent(self, intent):
        """ Capture OAuth redirect and extract token """
        data = intent.getDataString()
        if data and "access_token=" in data:
            token = self.extract_token_from_url(data)
            if token:
                self.fetch_user_info(token)

    def extract_token_from_url(self, url):
        """ Extracts the access token from the redirected URL """
        match = re.search(r"access_token=([^&]+)", url)
        if match:
            return match.group(1)
        return None

    def fetch_user_info(self, token):
        """ Fetch user profile using Google OAuth token """
        url = "https://www.googleapis.com/oauth2/v1/userinfo"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            user_info = response.json()
            self.update_ui(user_info)
        else:
            self.user_info_label.text = "Login failed. Try again."

    @mainthread
    def update_ui(self, user_info):
        """ Update UI with user info """
        self.user_info_label.text = f"Hello, {user_info['name']}!"
        self.profile_image.source = user_info["picture"]

class MyApp(App):
    def build(self):
        return LoginScreen()

if __name__ == "__main__":
    MyApp().run()
