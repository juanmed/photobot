from rpi_hardware_pwm import HardwarePWM
from time import sleep

# Using RPI5 so channel=2 for gpio18
print("Setup HW PWM")
pwm = HardwarePWM(pwm_channel=2, hz=20_000, chip=0)
pwm.start(5) # full duty cycle
print("60hz, 95%")
sleep(5)
pwm.change_duty_cycle(10)
#pwm.change_frequency(25_000)
print("25khz, 50%")
sleep(5)
pwm.stop()
