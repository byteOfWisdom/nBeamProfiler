from zaber_motion import Units
from zaber_motion.ascii import Connection
import RPi.GPIO as GPIO
from sys import argv

max_speed = 10.
scan_speed = 2.5
sync_pin = 26

def make_scan_path(xsize, ysize, line_count):
    command_list = []
    #command_list.append((0, 0, max_speed))

    fwd = True
    for i in range(line_count):
        command_list.append((xsize if fwd else 0, (ysize / line_count) * i, scan_speed)) # the actual scan line
        command_list.append(((xsize if fwd else 0, (ysize / line_count) * (i + 1), max_speed))) # the height diff
        fwd = not fwd # switch direction for each row

    command_list[-1] = command_list[-2] # make the last command not move but just stay in place and send a pulse
    return command_list# [:-1] # don't need the last vertical step


if __name__ == "__main__":
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(sync_pin, GPIO.OUT)
    # requies argv: serialport, linecount, xsize (in cm), ysize (in cm)

    scan_cmds = make_scan_path(float(argv[3]) if len(argv) > 3 else 30., float(argv[4]) if len(argv) > 4 else 30., int(argv[2]) if len(argv) > 2 else 20)

    move_speed = float(argv[5]) if len(argv) > 5 else 2.5

    with Connection.open_serial_port(argv[1]) as connection:
        connection.enable_alerts()

        device_list = connection.detect_devices()
        print("Found {} devices".format(len(device_list)))

        #device = device_list[0]

        xaxis = device_list[0].get_axis(1)
        yaxis = device_list[1].get_axis(1)

        xaxis.settings.set("maxspeed", move_speed, Units.VELOCITY_CENTIMETRES_PER_SECOND)
        yaxis.settings.set("maxspeed", move_speed, Units.VELOCITY_CENTIMETRES_PER_SECOND)
        #if not xaxis.is_homed():
        xaxis.home()

        #if not yaxis.is_homed():
        yaxis.home()

        max_speed = yaxis.settings.get("maxspeed", Units.VELOCITY_CENTIMETRES_PER_SECOND)
        scan_speed = xaxis.settings.get("maxspeed", Units.VELOCITY_CENTIMETRES_PER_SECOND)

        i = 0
        for x, y, s in scan_cmds:
            GPIO.output(sync_pin, GPIO.HIGH)
            # xaxis.move_absolute(x * 10., Units.LENGTH_MILLIMETRES, velocity = scan_speed, velocity_unit = Units.VELOCITY_CENTIMETRES_PER_SECOND)
            xaxis.move_absolute(x * 10., Units.LENGTH_MILLIMETRES)
            GPIO.output(sync_pin, GPIO.LOW)
            # yaxis.move_absolute(y * 10., Units.LENGTH_MILLIMETRES, velocity = max_speed, velocity_unit = Units.VELOCITY_CENTIMETRES_PER_SECOND)
            yaxis.move_absolute(y * 10., Units.LENGTH_MILLIMETRES)
            print(f"finished move command {i} out of {len(scan_cmds)}")
            i += 1

    GPIO.cleanup()
