from flask import Blueprint

clothing_routes_bp = Blueprint("clothing_routes", __name__)


### General routes ###


# Get all the available clothing items
@clothing_routes_bp.route("/", methods=["GET"])
def getClothing():
    return BaseException(501, "Not Implemented")


# Create a new clothing item
@clothing_routes_bp.route("/", methods=["POST"])
def AddClothing():
    return BaseException(501, "Not Implemented")


# Delete a clothing item
@clothing_routes_bp.route("/", methods=["DELETE"])
def RemoveClothing():
    return BaseException(501, "Not Implemented")


### User specific routes ###


# Get all the available clothing items for a user
@clothing_routes_bp.route("/<string:username>", methods=["GET"])
def getClothingForUser(username):
    return BaseException(501, "Not Implemented")


# Create new clothing items for a user
@clothing_routes_bp.route("/<string:username>", methods=["POST"])
def addClothingForUser(username):
    return BaseException(501, "Not Implemented")


# Create new clothing items for a user
@clothing_routes_bp.route("/<string:username>", methods=["DELETE"])
def removeClothingForUser(username):
    return BaseException(501, "Not Implemented")
