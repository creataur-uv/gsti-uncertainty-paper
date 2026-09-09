import pysqlite3 as sqlite3 
# import sqlite3 
import pickle 
import helpers.timeUTC as time
import datetime


class db:
    def __init__(self, db_Name,table_Name):
        self.conn = sqlite3.connect(db_Name)
        self.cursor=self.conn.cursor()
        self.table_Name=table_Name
        self.cursor.execute(f"""CREATE TABLE IF NOT EXISTS {table_Name} (
id INTEGER PRIMARY KEY,
year TEXT NOT NULL,
month TEXT NOT NULL,
day TEXT NOT NULL,
hour TEXT NOT NULL,
minute TEXT NOT NULL,
second TEXT NOT NULL,
desc TEXT NOT NULL,
data BLOB NOT NULL)""")
    
    def insert(self, desc, data):
        time_Now = datetime.datetime.now()
        self.cursor.execute(f"""INSERT INTO {self.table_Name} (year, month, day, hour, minute, second, desc, data) VALUES (?,?,?,?,?,?, ?, ?)
""", (time_Now.year, time_Now.month, time_Now.day, time_Now.hour, time_Now.minute, time_Now.second, desc, pickle.dumps(data)))

    def closeDB(self):
        self.conn.commit()
        self.conn.close()


