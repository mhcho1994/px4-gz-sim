from pymavlink import mavutil

print("dialect module:", mavutil.mavlink.__name__)
print("EXTENDED_SYS_STATE =", mavutil.mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE)
print("FLIGHT_INFORMATION =", mavutil.mavlink.MAVLINK_MSG_ID_FLIGHT_INFORMATION)