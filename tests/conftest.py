import ipaddress
import socket

import pytest

from evalharness import runlog

_real_connect = socket.socket.connect


def _is_local(address) -> bool:
    if isinstance(address, str):  # unix socket path
        return True
    host = address[0]
    if host in ("localhost", "::1"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def no_outside_network(monkeypatch):
    """Tests never reach a real provider: only loopback connections are allowed."""
    def guarded(self, address):
        if not _is_local(address):
            raise RuntimeError(f"tests may not open network connections (tried {address})")
        return _real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture(autouse=True)
def idle_machine(monkeypatch):
    """Skip the real 5 s machine-load sample. Tests that need a load patch it themselves."""
    monkeypatch.setattr(runlog, "machine_load", lambda sample_s=5.0: runlog.MachineLoad(cpu_pct=1.0, gpu_pct=None))
