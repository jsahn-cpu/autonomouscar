/*
 * autodrive_ws - Arduino Mega Motor Controller
 * Board: Arduino Mega 2560
 * Motor driver model(s): DRI0042 x3 (confirmed -- same driver boards/pin
 * wiring reused from the earlier autovehi project, see
 * github.com/jsahn-cpu/autovehi/blob/main/arduino/mega_motor_controller/mega_motor_controller.ino).
 *
 * Physical layout: two independently-wired drive motors (left/right
 * wheels, each its own PWM+direction pins) plus a separate steering motor
 * for Ackermann-style front steering -- NOT a single combined drive motor.
 * Turning is done entirely by the steering motor; the ROS/software side
 * only ever sends ONE throttle value (see serial_protocol.py's
 * encode_drive), which this firmware applies identically to both left and
 * right drive motors (no differential wheel speeds -- not needed since
 * steering geometry is handled mechanically by the front steering motor,
 * not by driving the wheels at different speeds).
 *
 * Wire protocol (see autodrive_vehicle/core/serial_protocol.py -- keep the
 * two in sync if either changes):
 *   M <throttle_pwm>
 *     example: M 100
 *   ST <steer_pwm> <duration_ms>
 *     example: ST 120 150
 *     example: ST -120 150
 *   SC
 *     Reset software steering estimate to center
 *   STOP
 *
 * There is no real steering angle sensor yet, so steering is an open-loop
 * timed pulse: apply steer_pwm for duration_ms, then stop. steerEstimate is
 * a software-only accumulated guess used purely to clamp against a soft
 * travel limit -- it is not a measurement and will drift from the real
 * steering angle over time.
 */

// Corrected 2026-07-27 against the actual wiring -- the L/STEER/R pin
// assignments were rotated (each one pointed at a different physical
// motor than its name said), confirmed by commanding one motor at a time
// on the real vehicle and observing which one actually moved.
const int L_PWM = 4;
const int L_IN1 = 27;
const int L_IN2 = 26;

const int STEER_PWM = 2;
const int STEER_IN1 = 22;
const int STEER_IN2 = 23;

const int R_PWM = 3;
const int R_IN1 = 24;
const int R_IN2 = 25;

// Safety limits -- enforced here regardless of what the ROS side sends.
// TODO: confirm safe values once the drive/steering motors are characterized.
const int MAX_DRIVE_PWM = 180;
// Raised from 140 2026-07-27, then again after 220 was still reported weak
// -- now the hardware ceiling (analogWrite's own max), no artificial
// software cap left. Test in short pulses and watch for stalling/
// overheating while characterizing this further; lower this once a safe
// working value is found instead of leaving it maxed out long-term.
const int MAX_STEER_PWM = 255;
const int MAX_STEER_DURATION_MS = 250;

// Software-only steering estimate.
// This is NOT a real steering angle -- see file header.
// +-700 was an untested placeholder too, and repeated manual test pulses
// during characterization were hitting it (silently rejecting further
// steer commands in that direction) well before the real mechanical rack
// limit -- widened 2026-07-27 so it stops interfering with testing. Use
// the 'c' key in keyboard_teleop_node (sends SC) to reset the estimate to
// 0 if it drifts, rather than relying on this ceiling to catch it.
const int STEER_EST_MIN = -100000;
const int STEER_EST_MAX = 100000;
int steerEstimate = 0;

// Timed steering state
bool steeringActive = false;
unsigned long steeringStopTime = 0;

String inputLine = "";

void setup() {
  Serial.begin(115200);

  pinMode(L_PWM, OUTPUT);
  pinMode(L_IN1, OUTPUT);
  pinMode(L_IN2, OUTPUT);

  pinMode(R_PWM, OUTPUT);
  pinMode(R_IN1, OUTPUT);
  pinMode(R_IN2, OUTPUT);

  pinMode(STEER_PWM, OUTPUT);
  pinMode(STEER_IN1, OUTPUT);
  pinMode(STEER_IN2, OUTPUT);

  stopAll();

  Serial.println("READY");
}

void loop() {
  readSerialCommand();
  updateTimedSteering();
}

void readSerialCommand() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n') {
      parseCommand(inputLine);
      inputLine = "";
    } else {
      inputLine += c;
    }
  }
}

void parseCommand(String cmd) {
  cmd.trim();

  if (cmd.length() == 0) {
    return;
  }

  if (cmd == "STOP") {
    stopAll();
    Serial.println("OK STOP");
    return;
  }

  if (cmd == "SC") {
    steerEstimate = 0;
    stopSteering();
    Serial.println("OK SC 0");
    return;
  }

  // M throttle_pwm -- applied identically to both drive motors (see file
  // header for why there's no left/right differential here).
  if (cmd.startsWith("M ")) {
    int firstSpace = cmd.indexOf(' ');
    int throttlePwm = cmd.substring(firstSpace + 1).toInt();
    throttlePwm = constrain(throttlePwm, -MAX_DRIVE_PWM, MAX_DRIVE_PWM);

    setMotor(L_PWM, L_IN1, L_IN2, throttlePwm);
    setMotor(R_PWM, R_IN1, R_IN2, throttlePwm);

    Serial.print("OK M ");
    Serial.println(throttlePwm);
    return;
  }

  // ST steer_pwm duration_ms
  if (cmd.startsWith("ST ")) {
    int firstSpace = cmd.indexOf(' ');
    int secondSpace = cmd.indexOf(' ', firstSpace + 1);

    if (secondSpace < 0) {
      Serial.println("ERR ST FORMAT");
      return;
    }

    int steerPwm = cmd.substring(firstSpace + 1, secondSpace).toInt();
    int durationMs = cmd.substring(secondSpace + 1).toInt();

    timedSteering(steerPwm, durationMs);
    return;
  }

  Serial.print("ERR UNKNOWN ");
  Serial.println(cmd);
}

void timedSteering(int steerPwm, int durationMs) {
  steerPwm = constrain(steerPwm, -MAX_STEER_PWM, MAX_STEER_PWM);
  durationMs = constrain(durationMs, 0, MAX_STEER_DURATION_MS);

  if (steerPwm == 0 || durationMs == 0) {
    stopSteering();
    Serial.println("OK ST 0 0");
    return;
  }

  int direction = (steerPwm > 0) ? 1 : -1;
  int estimatedNext = steerEstimate + direction * durationMs;

  if (estimatedNext > STEER_EST_MAX) {
    Serial.println("ERR STEER RIGHT LIMIT");
    stopSteering();
    return;
  }

  if (estimatedNext < STEER_EST_MIN) {
    Serial.println("ERR STEER LEFT LIMIT");
    stopSteering();
    return;
  }

  steerEstimate = estimatedNext;

  setMotor(STEER_PWM, STEER_IN1, STEER_IN2, steerPwm);

  steeringActive = true;
  steeringStopTime = millis() + durationMs;

  Serial.print("OK ST ");
  Serial.print(steerPwm);
  Serial.print(" ");
  Serial.print(durationMs);
  Serial.print(" EST ");
  Serial.println(steerEstimate);
}

void updateTimedSteering() {
  if (!steeringActive) {
    return;
  }

  if ((long)(millis() - steeringStopTime) >= 0) {
    stopSteering();
    Serial.print("OK STEER DONE EST ");
    Serial.println(steerEstimate);
  }
}

void setMotor(int pwmPin, int in1Pin, int in2Pin, int pwmValue) {
  pwmValue = constrain(pwmValue, -255, 255);

  if (pwmValue > 0) {
    digitalWrite(in1Pin, HIGH);
    digitalWrite(in2Pin, LOW);
    analogWrite(pwmPin, pwmValue);
  } else if (pwmValue < 0) {
    digitalWrite(in1Pin, LOW);
    digitalWrite(in2Pin, HIGH);
    analogWrite(pwmPin, -pwmValue);
  } else {
    digitalWrite(in1Pin, LOW);
    digitalWrite(in2Pin, LOW);
    analogWrite(pwmPin, 0);
  }
}

void stopSteering() {
  setMotor(STEER_PWM, STEER_IN1, STEER_IN2, 0);
  steeringActive = false;
}

void stopDrive() {
  setMotor(L_PWM, L_IN1, L_IN2, 0);
  setMotor(R_PWM, R_IN1, R_IN2, 0);
}

void stopAll() {
  stopSteering();
  stopDrive();
}
