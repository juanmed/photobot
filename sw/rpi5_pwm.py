from rpi_hardware_pwm import HardwarePWM
from time import sleep

# Using RPI5 so channel=2 for gpio18
pwm = HardwarePWM(pwm_channel=2, hz=60, chip=0)
pwm.start(100) # full duty cycle
sleep(2)
pwm.change_duty_cycle(50)
pwm.change_frequency(25_000)
sleep(2)
pwm.stop()