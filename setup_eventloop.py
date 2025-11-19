"""
Run this BEFORE setup.py to configure the event loop
Command: python setup_eventloop.py
"""

import os
import sys

print("Configuring Cassandra driver for Python 3.12+...")
print(f"Python version: {sys.version}")

# Set environment variable
os.environ['CASSANDRA_DRIVER_EVENT_LOOP'] = 'asyncio'

# Test import
try:
    print("\nTesting cassandra-driver import...")
    from cassandra.cluster import Cluster
    print("✅ SUCCESS! Cassandra driver can be imported.")
    print("\nNow run: python setup.py")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("\n" + "="*60)
    print("FIX STEPS:")
    print("="*60)
    print("""
1. Uninstall current packages:
   pip uninstall cassandra-driver cassio -y

2. Install with asyncio support:
   pip install cassandra-driver[asyncio]
   pip install cassio

3. If still failing, install eventlet:
   pip install eventlet
   pip install cassandra-driver[eventlet]

4. Run this script again:
   python setup_eventloop.py
    """)
    sys.exit(1)