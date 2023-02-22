from flask import Flask, request, session
import logging
from flask_pymongo import PyMongo

app = Flask(__name__)


# Config
app.config.from_object('tool_app.config')

# views
import tool_app.views