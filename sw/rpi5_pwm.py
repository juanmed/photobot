from rpi_hardware_pwm import HardwarePWM
from time import sleep

# Using RPI5 so channel=2 for gpio18
print("Setup HW PWM @ 20khz")
pwm = HardwarePWM(pwm_channel=2, hz=20_000, chip=0)
pwm.start(0) # stop
sleep(1)

# Ramp duty cycle up
print("Ramp up!")
for i in range(0, 100, 5):    
    print(f"DT: {i}%")
    pwm.change_duty_cycle(i)
    sleep(1)

print("Ramp down!")
for i in range(100, 0, -5):
    range()    
    print(f"DT: {i}%")
    pwm.change_duty_cycle(i)
    sleep(1)

pwm.stop()
