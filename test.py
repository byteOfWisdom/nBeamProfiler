import RPi.GPIO as GPIO
from time import sleep
from sys import argv

sync_pin = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(sync_pin, GPIO.OUT)
delay = int(argv[2]) * 1e-3

for _ in range(int(argv[1])):
    GPIO.output(sync_pin, GPIO.HIGH)
    sleep(delay)
    GPIO.output(sync_pin, GPIO.LOW)
    sleep(delay)

GPIO.cleanup()
