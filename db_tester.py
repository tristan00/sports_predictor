import sqlite3

with sqlite3.connect('mma.db') as conn:

    res = conn.execute('''select * from matches''').fetchall()
    for i in res:
        print(i)
