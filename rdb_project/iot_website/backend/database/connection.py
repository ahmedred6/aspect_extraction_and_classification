import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv(".env")

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="iot"
    )
