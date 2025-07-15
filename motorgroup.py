import ev3dev2.motor
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, OUTPUT_C, SpeedPercent
from ev3dev2.sound import Sound

# Initialize motors (adjust OUTPUT_X to your wiring)
motor_recycle = LargeMotor(OUTPUT_A)
motor_compost = LargeMotor(OUTPUT_B)
motor_trash = LargeMotor(OUTPUT_C)
sound = Sound()