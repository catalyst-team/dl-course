import logging

from fastapi import FastAPI
from reaction.__version__ import __version__
from reaction.rpc import RabbitRPC

logging.basicConfig(level=logging.DEBUG)


class rpc(RabbitRPC):
    URL = "amqp://admin:8c85904bf645@queue"


app = FastAPI(debug=True, title="Reaction TeleBot Example", version=__version__)
