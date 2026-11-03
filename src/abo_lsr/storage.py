from __future__ import annotations
import sqlite3, json, time
from pathlib import Path

class ExperimentStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS evaluations(
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, step INTEGER,
            x_json TEXT, params_json TEXT, score REAL, status TEXT,
            gp_mean REAL, gp_std REAL, z REAL, belief_json TEXT,
            policy TEXT, elapsed REAL, created REAL)""")
        self.conn.commit()

    def add(self, run_id, step, x, params, score, status, gp_mean, gp_std,
            z, belief, policy, elapsed):
        self.conn.execute("""INSERT INTO evaluations
        (run_id,step,x_json,params_json,score,status,gp_mean,gp_std,z,belief_json,
         policy,elapsed,created) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id,step,json.dumps(x),json.dumps(params),score,status,gp_mean,gp_std,z,
         json.dumps(belief),policy,elapsed,time.time()))
        self.conn.commit()

    def close(self):
        self.conn.close()
