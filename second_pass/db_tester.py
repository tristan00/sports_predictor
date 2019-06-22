import sqlite3
import traceback

with sqlite3.connect('mma.db') as conn:

    res = conn.execute('''select * from fighter''').fetchall()

with sqlite3.connect('mma2.db') as conn:
    for i in res:
        try:
            conn.execute('''insert into  fighter values (?, ?, ?)''', (i[0], i[1], i[2]))
        except:
            pass

with sqlite3.connect('mma.db') as conn:

    res = conn.execute('''select * from matches''').fetchall()

with sqlite3.connect('mma2.db') as conn:
    for i in res:
        try:
            conn.execute('''insert into  matches values (?, ?, ?, ?, ?, ?,?)''', (i[0], i[1], i[2], i[3], i[4], i[5], i[6]))
        except:
            pass
