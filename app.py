from flask import Flask

from routes.clothing_routes import clothing_routes_bp
from routes.weather_routes import weather_routes_bp
from routes.llm_routes import llm_routes_bp

app = Flask(__name__)
app.register_blueprint(clothing_routes_bp, url_prefix="/clothing")
app.register_blueprint(weather_routes_bp, url_prefix="/weather")
app.register_blueprint(llm_routes_bp, url_prefix="/llm")


@app.route("/")
def hello_world():
    return "<h1>Hello, World!</h1>"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
