"""
Native function stubs for simulation.

These functions are stubs that simulate the behavior of native code
from the main Omerta project. In the actual implementation, these
would call into Swift code for VM connectivity checks.
"""

from .vm_connectivity import check_consumer_connected, check_vm_connectivity

__all__ = ["check_consumer_connected", "check_vm_connectivity"]
