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
        # Initialize the rotary encoder on GPIO pins 17(CLK) and 27(DT) with wrap-around at max_steps of 16
        self.encoder = RotaryEncoder(a=17, b=27, wrap=True, max_steps=16)
        # Initialize the rotary encoder's SW pin on GPIO pin 22
        self.button = Button(22)
    
    async def read_step(self)->int:
        return self.encoder.steps

async def main():
    # Using RPI5 so channel=2 for gpio18
    print("Setup HW PWM @ 20khz")
    pwm = HardwarePWM(pwm_channel=2, hz=20_000, chip=0)
    pwm.start(0) # stop
    sleep(1)

    encoder = Encoder()

    try:
        while True:
            await pwm_ramp(pwm)
            step = await encoder.read_step()
            print(f"Current step: {step}")
    finally:
        pwm.stop()
        print("Stopping pwm...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("User stopped")


    
