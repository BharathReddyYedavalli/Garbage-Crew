import time
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, OUTPUT_C, OUTPUT_D, SpeedPercent

class GarbageSortController:
    """Sequence: panels → rods → trap opens → trash drops → trap closes → rods retract → panels reset."""

    def __init__(self):
        try:
            self.panel_left = LargeMotor(OUTPUT_A)
            self.panel_right = LargeMotor(OUTPUT_B)
            self.rod_motor = LargeMotor(OUTPUT_C)
            self.trap_motor = LargeMotor(OUTPUT_D)
            self.motors_available = True
            print("Motors initialized.")
        except Exception as e:
            print(f"Motor init failed: {e}")
            self.motors_available = False

        self.speed = SpeedPercent(50)
        self.panel_deg = 45
        self.rod_deg = 40
        self.trap_deg = 95

    def shift_panels(self, direction):
        """45° shift: 'left', 'right', or None."""
        if not self.motors_available:
            print(f"Panels move {direction or 'none'} (simulated).")
            return
        deg = self.panel_deg if direction == 'left' else -self.panel_deg if direction == 'right' else 0
        if deg:
            self.panel_left.on_for_degrees(self.speed, deg, block=False)
            self.panel_right.on_for_degrees(self.speed, deg)
        time.sleep(0.2)

    def reset_panels(self):
        """Return panels to center."""
        if not self.motors_available:
            print("Panels reset to center.")
            return
        self.panel_left.on_for_degrees(self.speed, -self.panel_deg, block=False)
        self.panel_right.on_for_degrees(self.speed, -self.panel_deg)
        time.sleep(0.2)

    def extend_rods(self):
        """Raise support rods."""
        if not self.motors_available:
            print("Rods extend.")
            return
        self.rod_motor.on_for_degrees(self.speed, self.rod_deg)
        time.sleep(0.2)

    def retract_rods(self):
        """Lower rods."""
        if not self.motors_available:
            print("Rods retract.")
            return
        self.rod_motor.on_for_degrees(self.speed, -self.rod_deg)
        time.sleep(0.2)

    def open_trap(self):
        """Open trap by moving downward."""
        if not self.motors_available:
            print("Trap opens.")
            return
        self.trap_motor.on_for_degrees(self.speed, -self.trap_deg)
        time.sleep(0.2)

    def close_trap(self):
        """Close trap by moving upward."""
        if not self.motors_available:
            print("Trap closes.")
            return
        self.trap_motor.on_for_degrees(self.speed, self.trap_deg)
        time.sleep(0.2)

    def handle_classification(self, label):
        print(f"→ Classified: {label}")
        if label in ("glass", "textiles", "battery"):
            print("⚠️ Disallowed item: no movement.")
            return

        direction = {
            "trash": "right",
            "compost": None,
            "paper_cardboard": "left",
            "plastic": "left",
            "metal": "left"
        }.get(label)

        if direction is None and label not in ("compost", "trash"):
            print(f"Unknown label '{label}'; aborting.")
            return

        self.shift_panels(direction)
        self.extend_rods()
        self.open_trap()
        time.sleep(0.5)  # allow trash to fall
        self.close_trap()
        self.retract_rods()
        self.reset_panels()

    def set_motor_settings(self, speed=50, panel_deg=45, rod_deg=90, trap_deg=80):
        self.speed = SpeedPercent(speed)
        self.panel_deg = panel_deg
        self.rod_deg = rod_deg
        self.trap_deg = trap_deg
        print(f"[CONFIG] speed {speed}%, panel {panel_deg}°, rod {rod_deg}°, trap {trap_deg}°")

    def stop_all_motors(self):
        if self.motors_available:
            self.panel_left.stop()
            self.panel_right.stop()
            self.rod_motor.stop()
            self.trap_motor.stop()
        print("[STOP] All motors stopped.")

# Global instance
garbage_sort_controller = GarbageSortController()
