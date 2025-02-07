# # Copyright 2024 HARDPro <patrick.miller@gmail.com>

# import json
# import machine
# import network
# import select
# import socket
# import sys
# import ubluetooth
# import ubinascii
# import utime
# import re

# def shutdown(station):
#     station.disconnect()
#     station.active(False)
#     return

# # Our leds
# led = machine.Pin('LED', machine.Pin.OUT)
# LED0 = machine.Pin(0,machine.Pin.OUT)
# LED1 = machine.Pin(1,machine.Pin.OUT)
# LED2 = machine.Pin(2,machine.Pin.OUT)

# # Board mac address
# MAC = ubinascii.hexlify(network.WLAN().config('mac'),':').decode()
# print(MAC)

# led.on()
# LED0.off()
# LED1.off()

# def blink(n):
#     if not n: return
    
#     v = led.value()
#     interval = 1./n/2.

#     for _ in range(n):
#         led.off()
#         utime.sleep(interval)
#         led.on()
#         utime.sleep(interval)
#     led.off()
#     utime.sleep(interval)
#     if v:
#         led.on()
#     else:
#         led.off()
#     return

# class SocketCloser:
#     def __init__(self,s):
#         self.socket = s
#         return
#     def __enter__(self):
#         return self
#     def __exit__(self,*args):
#         self.socket.close()
#         return
    
# class SleepBlink:
#     def __init__(self,led,runtime,interval):
#         self.led = led
#         self.interval = interval
#         self.count = int(runtime/interval+0.5)
#         return
    
#     def __bool__(self):
#         if self.led.value():
#             self.led.off()
#         else:
#             self.led.on()
#         if self.count <= 0: return False
#         self.count -= 1
#         utime.sleep(self.interval)
#         return True   

# station = network.WLAN(network.STA_IF)

# try:
#     # Read config file
#     try:
#         with open('config.json') as c:
#             config = json.load(c)
#     except:
#         config = {}
    
#     print('Starting...',station)
#     if station.isconnected():
#         station.disconnect()
#     station.active(False) # Good to turn off the network before trying a new connection

#     station.active(True)
#     station.connect('inad','Dani cooks!')
#     sleeper = SleepBlink(led,5,.2)
#     while sleeper:
#         if station.isconnected(): break
#     if station.isconnected():
#         print('We have an INAD, get the current config')
#         ip = station.ipconfig('addr4')[0]
#         print('up at',ip)
#         inad = '.'.join(ip.split('.')[:3]+['1'])
#         print('inad at',inad)
        
#         # Open a socket to inad:80 and send a HTTP GET request
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         with SocketCloser(s):
#             # Send a HTTP request
#             s.connect((inad,80))
#             s.send(b'GET /config HTTP/1.0\r\n')
            
#             # Read the response which includes a HTTP response header
#             msg = b''
#             while True:
#                 b = s.recv(1024)
#                 if not b: break
#                 msg += b
#         print(msg)
#         match = re.search(b'\r\n\r\n',msg)
#         if match is not None:
#             con = msg[match.end():]
#             new_config = json.loads(con)
#             with open('config.json','wb') as c:
#                 json.dump(new_config,c)
#             config = new_config
#             print('new config')
#             print(config)
            
#     # OK, now our config is set (and maybe updated from an INAD)
#     # We will make an effort to connect to the WiFi from the config
#     # (stopping our connection to the INAD if we have one).  If we don't
#     # get one, we will start advertising on Bluetooth anyway, but we
#     # won't know our server address until we get the ping.
#     shutdown(station)
#     station.active(True)
#     print('connect to',config['ssid'],config['password'])
#     station.connect(config['ssid'],config['password'])
 
#  # Turn on bluetooth and so we can start advertising
#     ble = ubluetooth.BLE()
#     ble.active(True)
    
#     UDPPORT = config['udp']
#     ANCHORPORT = config['anchor']
#     SECRET = bytes(config['secret'],'utf8')
    
#     wifi = False
#     client = None
#     anchor = None
#     server = 0
#     identifier = 0
#     last_ad = None
    
#     # Turn on lights for wifi and fully connected
#     led.off()
#     LED0.off()

#     while 1:
#         #print(wifi,server_ping,advertising)
#         #print('client is',client)
        
#         # The blink rate will tell us what has worked
#         #blink(1+wifi+server_ping+advertising)
        
#         # Did our expectd wifi connection come up?  If it did, start looking
#         # for a ping
#         if not wifi and station.isconnected():
#             print('Connected to wifi from config:',config['ssid'])
#             wifi = True
#             led.on()
            
#             client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#             client.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
#             client.bind(("", UDPPORT))
#             print('UDP listener at',UDPPORT)
            
#         # If we have a client socket waiting for a ping from the server, we attempt to
#         # read from it
#         if client is not None:
#             ready = select.select([client],[],[],.1)
#             if ready[0]:
#                 print('saw a ping',UDPPORT)
#                 message,addr = client.recvfrom(1024)
#                 print("heard ping",message,"from",addr)
#                 if message == SECRET:
#                     server_ip = addr[0]
#                     server = sum([int(x)*y for x,y in zip(server_ip.split('.'),(2**24,2**16,2**8,2**1))])
#                     print('server',server)
#                     client.close()
#                     client = None
                    
#                     print('waiting for board identification from',server_ip,ANCHORPORT)
#                     anchor = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
#                     anchor.connect((server_ip,ANCHORPORT))
#                     anchor.send(bytes(MAC,'utf8')+b'\r\n')

#         # We need the server to tell us our point id and other info
#         if anchor is not None:
#             ready = select.select([anchor],[],[],.1)
#             if ready[0]:
#                 response = anchor.recv(1024)
#                 print('anchor socket got',response)
#                 anchor.close()
#                 anchor = None
#                 info = json.loads(response.decode('utf8'))
#                 print(info)
#                 identifier = info.get('id')
#                 LED0.on()
                
#         # We'll advertise on bluetooth.  The info will get better as we first get the server number
#         # and then get the board number
#         if ble.active():
#             ADV_DATA_FORMAT = b'\x02\x01\x06\x16\tstrumbeatz%03d%08x\x03\x19\x00\x00'
#             ad = ADV_DATA_FORMAT%(identifier,server)
#             if ad != last_ad:
#                 print(ad)
#                 last_ad = ad
#             ble.gap_advertise(300,adv_data=bytearray(ad))
# except:
#     # Should turn on an error indicator to show there was an abort
#     LED1.on()
#     shutdown(station)
#     raise

# shutdown(station)# Compatibility layer for MicroPython modules





#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################

# Mock implementations for MicroPython-specific modules
class MockPin:
    OUT = 'OUT'
    IN = 'IN'
    
    def __init__(self, pin, mode=None):
        self._pin = pin
        self._mode = mode
        self._value = False
    
    def on(self):
        self._value = True
    
    def off(self):
        self._value = False
    
    def value(self, val=None):
        if val is not None:
            self._value = bool(val)
        return self._value

class MockNetwork:
    STA_IF = 'station'
    
    def __init__(self, interface_type):
        self._type = interface_type
        self._connected = False
        self._active = False
    
    def active(self, state=None):
        if state is not None:
            self._active = state
        return self._active
    
    def connect(self, ssid, password):
        print(f"Connecting to {ssid}")
        self._connected = True
    
    def isconnected(self):
        return self._connected
    
    def disconnect(self):
        self._connected = False
    
    def config(self, param):
        if param == 'mac':
            return b'\x00\x11\x22\x33\x44\x55'  # Mock MAC address
    
    def ipconfig(self, param):
        if param == 'addr4':
            return ['192.168.1.100']

class MockBluetooth:
    def __init__(self):
        self._active = False
    
    def active(self, state=None):
        if state is not None:
            self._active = state
        return self._active
    
    def gap_advertise(self, interval, adv_data=None):
        print(f"Advertising: {adv_data}")

# Replace MicroPython-specific imports with mock implementations
import json
import socket
import select
import sys
import time
import re
import binascii

# Mock machine module
class machine:
    Pin = MockPin
    
# Mock network module    
class network:
    WLAN = MockNetwork
    STA_IF = 'station'

# Mock ubluetooth module
class ubluetooth:
    BLE = MockBluetooth

# Mock ubinascii module
class ubinascii:
    @staticmethod
    def hexlify(data, separator=b':'):
        return binascii.hexlify(data).decode()

# Replace utime with time
utime = time

# Rest of the original script (with minor modifications)
def shutdown(station):
    station.disconnect()
    station.active(False)
    return

# Our leds
led = machine.Pin('LED', machine.Pin.OUT)
LED0 = machine.Pin(0, machine.Pin.OUT)
LED1 = machine.Pin(1, machine.Pin.OUT)
LED2 = machine.Pin(2, machine.Pin.OUT)

# Board mac address
MAC = ubinascii.hexlify(network.WLAN().config('mac'), ':')
print(MAC)

led.on()
LED0.off()
LED1.off()

def blink(n):
    if not n: return
    
    v = led.value()
    interval = 1./n/2.

    for _ in range(n):
        led.off()
        utime.sleep(interval)
        led.on()
        utime.sleep(interval)
    led.off()
    utime.sleep(interval)
    if v:
        led.on()
    else:
        led.off()
    return

class SocketCloser:
    def __init__(self, s):
        self.socket = s
        return
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.socket.close()
        return
    
class SleepBlink:
    def __init__(self, led, runtime, interval):
        self.led = led
        self.interval = interval
        self.count = int(runtime/interval+0.5)
        return
    
    def __bool__(self):
        if self.led.value():
            self.led.off()
        else:
            self.led.on()
        if self.count <= 0: return False
        self.count -= 1
        utime.sleep(self.interval)
        return True   

station = network.WLAN(network.STA_IF)

try:
    # Read config file
    try:
        with open('config.json') as c:
            config = json.load(c)
    except Exception as e:
        print(f"Could not load config: {e}")
        config = {
            'ssid': 'inad',
            'password': 'Dani cooks!',
            'udp': 12345,
            'anchor': 54321,
            'secret': 'test_secret'
        }
    
    print('Starting...', station)
    if station.isconnected():
        station.disconnect()
    station.active(False)

    station.active(True)
    station.connect(config['ssid'], config['password'])
    sleeper = SleepBlink(led, 5, .2)
    while sleeper:
        if station.isconnected(): break
    
    if station.isconnected():
        print('Connected to network')
        # Simulate the rest of the connection and configuration process
        print('Simulating network and Bluetooth setup')
        
        # Simulate Bluetooth advertising
        ble = ubluetooth.BLE()
        ble.active(True)
        
        # Print out configuration for demonstration
        print("Configuration:", config)

except Exception as e:
    # Error handling
    LED1.on()
    shutdown(station)
    print(f"An error occurred: {e}")
    raise

finally:
    shutdown(station)

print("Script execution completed.")