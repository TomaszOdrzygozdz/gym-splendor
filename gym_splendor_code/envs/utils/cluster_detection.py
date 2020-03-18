import os
host_name = os.popen('hostname').read()
CLUSTER = not host_name == 'tomasz-LAPTOP\n'
