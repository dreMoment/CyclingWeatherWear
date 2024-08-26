from flask import Blueprint

weather_routes_bp = Blueprint("weather_routes", __name__)


# Get the weather for a specific location
@weather_routes_bp.route("/<string:location>", methods=["GET"])
def getWeather(location):
    return BaseException(501, "Not Implemented")
