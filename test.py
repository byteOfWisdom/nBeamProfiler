import RPi.GPIO as GPIO
from time import sleep

sync_pin = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(sync_pin, GPIO.OUT)

while 1:
    GPIO.output(sync_pin, GPIO.HIGH)
    sleep(0.1)
    GPIO.output(sync_pin, GPIO.LOW)
    sleep(0.1)
