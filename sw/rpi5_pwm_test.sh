#!/bin/bash
NODE=/sys/class/pwm/pwmchip1
PIN="$1"
FUNC="a0"
PERIOD="$2"
DUTY_CYCLE="$3"

function usage {
	printf "Usage: $0 <channel> <period> <duty_cycle>\n"
	printf "    pin - one of 12, 13, 14, 15, 18 or 19\n"
	printf "    period - PWM period in nanoseconds\n"
	printf "    duty_cycle - Duty Cycle (on period) in nanoseconds\n"
	exit 1
}

if [ -d "$NODE/device/consumer:platform:cooling_fan/" ]; then
	echo "Hold your horses, looks like this is pwm1?"
	exit 1
fi

case $PIN in
	"12")
	CHANNEL="0"
	;;
	"13")
	CHANNEL="1"
	;;
	"14")
	CHANNEL="2"
	;;
	"15")
	CHANNEL="3"
	;;
	"18")
	CHANNEL="2"
	FUNC="a3"
	;;
	"19")
	CHANNEL="3"
	FUNC="a3"
	;;
	*)
	echo "Unknown pin $PIN."
	exit 1
esac

function pwmset {
	echo "$2" | sudo tee -a "$NODE/$1" > /dev/null
}

if [[ "$PERIOD" == "off" ]]; then	
	if [ -d "$NODE/pwm$CHANNEL" ]; then
		pinctrl set $PIN no
		pwmset "pwm$CHANNEL/enable" "0"
		pwmset "unexport" "$CHANNEL"
	fi
	exit 0
fi

if [[ ! $PERIOD =~ ^[0-9]+$ ]]; then
	usage
fi

if [[ ! $DUTY_CYCLE =~ ^[0-9]+$ ]]; then
	usage
fi

if [ ! -d "$NODE/pwm$CHANNEL" ]; then
	pwmset "export" "$CHANNEL"
fi

pwmset "pwm$CHANNEL/enable" "0"
pwmset "pwm$CHANNEL/period" "$PERIOD"
if [ $? -ne 0 ]; then
	echo "^ don't worry, handling it!"
	pwmset "pwm$CHANNEL/duty_cycle" "$DUTY_CYCLE"
	pwmset "pwm$CHANNEL/period" "$PERIOD"
else
	pwmset "pwm$CHANNEL/duty_cycle" "$DUTY_CYCLE"
fi
pwmset "pwm$CHANNEL/enable" "1"

# Sure, the pin is set to the correct alt mode by the dtoverlay at startup...
# But we'll do this to protect the user from themselves:
pinctrl set $PIN $FUNC
