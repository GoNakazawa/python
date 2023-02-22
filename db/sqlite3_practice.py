# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:49:20 2021

@author: 13191
"""
import sqlite3

con = sqlite3.connect("sample.db")
cursor = con.cursor()

create_user_table_query = """
CREATE TABLE IF NOT EXISTS User (
    UserId INTEGER PRIMARY KEY NOT NULL,
    Name TEXT DEFAULT 'anonymous',
    Email TEXT NOT NULL,
    Age INTEGER CHECK(Age > 0)
) 
"""

cursor.execute(create_user_table_query)

insert_user_query = "INSERT INTO User VALUES(1, 'John', 'john@gmail.com', 30)"
cursor.execute(insert_user_query)
con.commit()