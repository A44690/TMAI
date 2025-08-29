import time
from threading import Lock, Thread

from tminterface.client import Client
from tminterface.structs import SimStateData
from tminterface.interface import TMInterface


class SimStateClient(Client):
    """
    Client for a TMInterface instance.
    Its only job is to get the simulation state that is used by the gym env for reward computation.
    """

    def __init__(self):
        super().__init__()
        self.sim_state = SimStateData()

    def on_run_step(self, iface, _time: int):
        self.sim_state = iface.get_simulation_state()


class ThreadedClient:
    """
    Allows to run the Client in a separate thread, so that the gym env can run in the main thread.
    """

    def __init__(self):
        self.iface = TMInterface()
        self.tmi_client = SimStateClient()
        self._client_thread = Thread(target=self.client_thread, daemon=True)
        self._lock = Lock()
        self.iface.register(self.tmi_client)
        time.sleep(0)
        self._lock.acquire()
        self.data = self.tmi_client.sim_state
        self._lock.release()

    def client_thread(self):
        self.tmi_client = SimStateClient()
        print("ok")

        self.iface.register(self.tmi_client)
        while self.iface.running:
            time.sleep(0)
            self._lock.acquire()
            self.data = self.tmi_client.sim_state
            self._lock.release()
            
    def update(self):
        time.sleep(0)
        self._lock.acquire()
        self.data = self.tmi_client.sim_state
        self._lock.release()


if __name__ == "__main__":
    simthread = ThreadedClient()