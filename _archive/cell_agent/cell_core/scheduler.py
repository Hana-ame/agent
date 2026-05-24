import time
import json
from typing import Optional

from .db import get_connection, DB_PATH
from .dna import DNAExecutor


class Scheduler:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(DB_PATH)
        self.executor = DNAExecutor(self.db_path)
        self._running = False

    def poll_once(self):
        conn = get_connection(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id FROM tasks WHERE status = 'pending' ORDER BY id ASC LIMIT 5")
        tasks = cur.fetchall()
        conn.close()

        for (task_id,) in tasks:
            print(f"[Scheduler] Processing task #{task_id}")
            try:
                result = self.executor.run_task(task_id)
                print(f"[Scheduler] Task #{task_id} done: {result['status']}")
            except Exception as e:
                print(f"[Scheduler] Task #{task_id} failed: {e}")
                conn = get_connection(self.db_path)
                conn.execute("UPDATE tasks SET status = 'failed', error = ? WHERE id = ?",
                             (str(e)[:500], task_id))
                conn.commit()
                conn.close()

    def run_forever(self, interval: float = 2.0):
        self._running = True
        print(f"[Scheduler] Started (poll interval={interval}s)")
        while self._running:
            self.poll_once()
            time.sleep(interval)

    def stop(self):
        self._running = False


if __name__ == "__main__":
    from .db import init_db, seed_default_data
    init_db()
    seed_default_data()
    sched = Scheduler()
    sched.run_forever()
