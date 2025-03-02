import serial
import time

# Open the serial connection (make sure to replace 'COM_PORT' with your actual port)
ser = serial.Serial('COM8', 9600, timeout=1)

# Allow Arduino to reset
time.sleep(2)

# Send the unlock command to Arduino
print("Sending unlock command...")
ser.write(b'UNLOCK')  # Send the "UNLOCK" command

# Optionally, you can print a confirmation in the Python console
print("Unlock command sent!")

# Close the serial connection
ser.close()


#learning how to communicate with arduino through python and serial library. 