import serial

s = serial.Serial("/dev/rfcomm0", 115200, timeout=1)
print("OK:", s.name)
s.close()
