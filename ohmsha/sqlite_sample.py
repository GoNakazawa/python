# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 09:36:12 2021

@author: 13191
"""

import sqlite3
 
# 接続。なければDBを作成する。
conn = sqlite3.connect('example.db')
 
# カーソルを取得
c = conn.cursor()
 
# テーブルを作成
c.execute('CREATE TABLE articles  (id int, title varchar(1024), body text, created datetime)')
 
# Insert実行
c.execute("INSERT INTO articles VALUES (1,'今朝のおかず','魚を食べました','2020-02-01 00:00:00')")
c.execute("INSERT INTO articles VALUES (2,'今日のお昼ごはん','カレーを食べました','2020-02-02 00:00:00')")
c.execute("INSERT INTO articles VALUES (3,'今夜の夕食','夕食はハンバーグでした','2020-02-03 00:00:00')")
 
# コミット
conn.commit()
 
# コネクションをクローズ
conn.close()