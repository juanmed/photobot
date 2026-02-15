# Inspired from https://docs.sunfounder.com/projects/umsk/en/latest/05_raspberry_pi/pi_lesson17_rotary_encoder.html

from rpi_hardware_pwm import HardwarePWM
from time import sleep
import asyncio
from gpiozero import RotaryEncoder, Button

async def pwm_ramp(pwm: HardwarePWM) -> None:
    # Ramp duty cycle up
    print("Ramp up!")
    for i in range(0, 20, 5):    
        print(f"DT: {i}%")
        pwm.change_duty_cycle(i)
        await asyncio.sleep(1)

    print("Ramp down!")
    for i in range(20, 0, -5):   
        print(f"DT: {i}%")
        pwm.change_duty_cycle(i)
        await asyncio.sleep(1)

class Encoder:

    def __init__(self):
        # Initialize the rotary encoder on GPIO pins 17(CLK) and 27(DT) with wrap-around=0 
        # to let it count infinitely
        self.encoder = RotaryEncoder(a=17, b=27, wrap=False, max_steps=0)
        # Initialize the rotary encoder's SW pin on GPIO pin 22
        self.button = Button(22)

    async def read_step(self)->None:
        print(f"Current step: {self.encoder.steps}")
        await asyncio.sleep(0.05)
    
    async def read_button(self)->None:
        if self.button.is_pressed:
            print("Button pressed!")  # Print message on button press
            await asyncio.sleep(0.1)


async def main():
    # Using RPI5 so channel=2 for gpio18
    print("Setup HW PWM @ 20khz")
    pwm = HardwarePWM(pwm_channel=2, hz=20_000, chip=0)
    pwm.start(0) # stop
    sleep(1)

    encoder = Encoder()

    try:
        while True:
            t1 = asyncio.create_task(pwm_ramp(pwm))
            t2 = asyncio.create_task(encoder.read_step())
            t3 = asyncio.create_task(encoder.read_button())
            await t1
            await t2
            await t3
    finally:
        pwm.stop()
        print("Stopping pwm...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("User stopped")


    
