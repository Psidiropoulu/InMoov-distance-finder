import serial

def connect_lidar(port, baudrate):
    try:
        lidar = serial.Serial(port, baudrate)
        print("LiDAR connected successfully!")
        return lidar
    except serial.SerialException as e:
        print(f"Failed to connect to LiDAR: {e}")
        return None

# Example usage
lidar_port = "/dev/ttyUSB0"  # Replace with the actual port of your LiDAR
lidar_baudrate = 115200  # Replace with the appropriate baudrate

lidar = connect_lidar(lidar_port, lidar_baudrate)
if lidar is not None:
    # Perform LiDAR operations here
    lidar.close()
    
    
    def get_distance(lidar):
        try:
            # Send command to start measurement
            lidar.write(b'\x42\x57\x02\x00\x00\x00\x01\x00\x20\x11')

            # Read response
            response = lidar.read(9)

            # Extract distance from response
            distance = int.from_bytes(response[2:4], byteorder='little')

            return distance
        except serial.SerialException as e:
            print(f"Failed to get distance: {e}")
            return None

# Example usage
distance = get_distance(lidar)
if distance is not None:
    print(f"Distance: {distance} cm")
