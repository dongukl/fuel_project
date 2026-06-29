import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/gwon/Desktop/rokey_F4/install/fuel_port_perception'
