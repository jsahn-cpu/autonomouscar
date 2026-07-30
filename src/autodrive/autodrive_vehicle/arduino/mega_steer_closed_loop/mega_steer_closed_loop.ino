/*
 * mega_steer_closed_loop.ino — 가변저항(POT, A6) 피드백 기반 폐루프 조향 펌웨어
 * Board: Arduino Mega 2560 / 드라이버: DRI0042 x3 (기존 mega_motor_controller.ino와 동일 배선)
 *
 * mega_motor_controller.ino(open-loop 타임드 펄스)의 대체 버전이다. 차이는 조향뿐:
 *   - 예전: ST <pwm> <duration> 을 쏘고 소프트 추정치로만 각을 짐작 (측정 없음)
 *   - 지금: SA <target> 로 목표 ADC를 주면, 펌웨어가 매 루프 analogRead(A6)로 실제
 *           위치를 읽어 P제어로 그 목표를 "물고 유지"한다. 진짜 폐루프.
 *
 * 구동(주행) 모터는 예전과 동일 — M <pwm> 하나를 좌우 구동모터에 똑같이 적용.
 *
 * ── 캘리브레이션 (끝단 저항값이 물리적으로 드리프트하므로 재측정 가능해야 함) ──
 *   CAL 명령 하나로 좌/우 끝단을 자동 스윕 측정 → min/max/방향부호를 EEPROM에 저장.
 *   전원을 껐다 켜도 유지되고, 재캘은 CAL 재전송뿐 — 스케치 재업로드 불필요.
 *   측정된 사용자 값(우회전 210 / 좌회전 710, ADC 증가=좌)을 부팅 기본값으로 넣어둔다.
 *
 * ── 시리얼 프로토콜 (115200) ─────────────────────────────────────
 *   M <pwm>            구동 PWM (좌우 동일). 예: M 100
 *   SA <target_adc>    조향 목표 ADC (held). 소프트리밋으로 클램프. 예: SA 460
 *   SC                 목표를 중앙(center)으로
 *   SH                 조향 릴리스(모터 off, 폐루프 해제)
 *   SR <pwm>           [진단] 폐루프 없이 조향모터 생짜 구동 (SR 255 / SR -255)
 *   CAL                끝단 자동 스윕 → EEPROM 저장 → 중앙 복귀 후 hold
 *   CFG                현재 캘리브/게인 출력
 *   FBON / FBOFF       FB 텔레메트리 스트림 켜기/끄기 (CAL 중엔 자동 억제, CAL 끝나면 자동 off)
 *   STOP               전부 정지 + 조향 해제
 *
 * ── 텔레메트리 (50Hz) ────────────────────────────────────────────
 *   FB <pot> <target> <steer_pwm> <engaged>
 *     ROS steering_feedback_node가 <pot>을 rad로 변환해 발행한다.
 *   OK.../ERR.../EVENT... 은 명령 응답·상태 전이.
 *
 * ⚠️ 방향 부호(어느 극성이 ADC를 올리나)는 CAL로 확정된다. CAL 이력이 없으면 첫 SA에서
 *    짧게 nudge해 스스로 부호를 찾는다(부호 오류로 인한 폭주 방지). 소프트리밋+스톨감지가 백업.
 */

#include <EEPROM.h>

// ── 핀 (사용자 확정 배선, steer_pot_test.ino와 동일) ──
const int STEER_PWM = 2, STEER_IN1 = 22, STEER_IN2 = 23;
const int S_POT     = A6;
const int L_PWM = 4, L_IN1 = 27, L_IN2 = 26;   // mega_motor_controller.ino 확정 배선
const int R_PWM = 3, R_IN1 = 24, R_IN2 = 25;

// ── 안전/구동 한계 ──
const int MAX_DRIVE_PWM = 180;

// ── 조향 폐루프 파라미터 (튜닝 대상) ──
const int STEER_SWEEP_PWM = 255;   // CAL 스윕/probe 구동 (약한 방향 확인 위해 최대)
// 조향 폐루프 = 펄스-정착(pulse & settle). 모터 켠 채 pot을 읽으면 값이 튀어(모터 노이즈)
// 목표 근처에서 헌팅한다 -> 목표 근처(NEAR_ZONE)에선 짧게 밀고(펄스) 모터 끄고 깨끗이
// 읽어(정착) 판단하기를 반복. 멀면 연속 구동(빠름). 비백드라이브 기어박스라 모터 꺼도
// 위치가 유지되므로 능동 홀드가 필요 없다.
const int STEER_DRIVE_PWM = 255;              // 구동 PWM (무게 증가+한쪽 약함 -> 최대 255로 테스트)
const int NEAR_ZONE       = 90;               // 이 밖은 긴 펄스(빠름), 안쪽은 짧은 펄스(정밀)
const unsigned long HOLD_PULSE_FAR_MS  = 200; // 원거리 펄스 길이
const unsigned long HOLD_PULSE_NEAR_MS = 50;  // 근거리 펄스 길이
const unsigned long HOLD_SETTLE_MS     = 70;  // 끄고 깨끗이 읽는 시간
const int STEER_DEADBAND  = 20;               // 목표 근처 정지대 (깨끗한 읽기 기준)
const int LIMIT_MARGIN    = 15;               // 소프트리밋을 끝단에서 이만큼 안쪽으로

// ── 캘리브레이션(끝단) 감지 파라미터 ──
const unsigned long STALL_WINDOW_MS = 600;
const int           STALL_MIN_DELTA = 3;
// 순간적으로 느려진 걸 스톨로 오판하지 않도록, 이만큼 "연속" 창에서 안 움직여야 스톨로 확정.
// (예: 3 x 600ms = 1.8초 동안 계속 <3카운트여야 진짜 끝단/막힘으로 판정)
const int           STALL_WINDOWS_NEEDED = 3;
const unsigned long MAX_PHASE_MS    = 6000;
const unsigned long SETTLE_MS       = 400;
const unsigned long PROBE_MS        = 300;   // 방향부호 nudge 시간
const unsigned long REPORT_MS       = 20;

// ── EEPROM 저장 구조 ──
struct Calib { uint8_t magic; int potMin; int potMax; int8_t increaseA; };
const uint8_t CAL_MAGIC = 0xA6;
Calib cal = { 0, 210, 710, -1 };   // 기본: 우210/좌710, 부호 미상(-1)

int center() { return (cal.potMin + cal.potMax) / 2; }
int loLimit() { return cal.potMin + LIMIT_MARGIN; }
int hiLimit() { return cal.potMax - LIMIT_MARGIN; }

// ── 상태 ──
enum Mode { M_IDLE, M_HOLD, M_PROBE, M_RAW, M_CAL_A, M_CAL_SA, M_CAL_B, M_CAL_SB, M_CAL_CENTER };
Mode mode = M_IDLE;
int rawSteer = 0;   // SR 명령: 폐루프 없이 조향모터를 이 PWM으로 생짜 구동(진단용)

int  steerTarget = -1;        // 현재 조향 목표 ADC (-1 = 미설정)
int  steerPwmOut = 0;         // 마지막 조향 출력 (텔레메트리용)
bool stalledAtTarget = false; // 목표 추종 중 기계 끝/막힘으로 정지한 상태
bool signFlipTried = false;   // 목표당 부호 자동뒤집기 1회 제한(핑퐁 방지)
bool telemetry = true;        // FB 스트림 on/off. CAL 중엔 자동 억제, CAL 끝나면 자동 off
                              // (Serial Monitor에서 SUMMARY를 읽을 수 있게). 구동명령/FBON으로 재개.
bool holdDriving = false;     // 펄스-정착: 지금 미는 중(true)인지 정착/읽기 중(false)인지
int  holdDir = 0;            // 펄스 시작 시 고정한 구동 방향(+1/-1)
unsigned long tHoldStep = 0; // 현재 펄스/정착 시작 시각
unsigned long holdPulseMs = HOLD_PULSE_NEAR_MS; // 이번 펄스 길이(원거리=길게, 근거리=짧게)

unsigned long tPhaseStart = 0, tReport = 0, tStallRef = 0, tSettle = 0, tProbe = 0, tDirRef = 0;
int potAtStallRef = 0, potAtProbe = 0, potAtDirRef = 0, potPhaseStart = 0;
int stallWindows = 0;   // 연속으로 안 움직인 창 개수 (STALL_WINDOWS_NEEDED 도달 시 스톨 확정)
int calEndA = -1, calEndB = -1;
int calSweepPwm = STEER_SWEEP_PWM;
const int MIN_SWEEP_TRAVEL = 40;   // CAL: 이만큼도 안 움직인 스톨은 false stall로 보고 무시

int medBuf[5];
int medIdx = 0;
String inputLine = "";

int median5() {
  int a[5];
  for (int i = 0; i < 5; i++) a[i] = medBuf[i];
  for (int i = 1; i < 5; i++) { int k=a[i], j=i-1; while (j>=0 && a[j]>k){a[j+1]=a[j];j--;} a[j+1]=k; }
  return a[2];
}

void setSteerMotor(int pwm) {   // +pwm = S_IN1 HIGH(=극성 A), -pwm = S_IN2 HIGH(=극성 B)
  pwm = constrain(pwm, -255, 255);
  if (pwm > 0)      { digitalWrite(STEER_IN1, HIGH); digitalWrite(STEER_IN2, LOW);  analogWrite(STEER_PWM, pwm); }
  else if (pwm < 0) { digitalWrite(STEER_IN1, LOW);  digitalWrite(STEER_IN2, HIGH); analogWrite(STEER_PWM, -pwm); }
  else              { digitalWrite(STEER_IN1, LOW);  digitalWrite(STEER_IN2, LOW);  analogWrite(STEER_PWM, 0); }
}

void setDriveMotors(int pwm) {
  pwm = constrain(pwm, -MAX_DRIVE_PWM, MAX_DRIVE_PWM);
  // L
  if (pwm > 0)      { digitalWrite(L_IN1, HIGH); digitalWrite(L_IN2, LOW);  analogWrite(L_PWM, pwm); }
  else if (pwm < 0) { digitalWrite(L_IN1, LOW);  digitalWrite(L_IN2, HIGH); analogWrite(L_PWM, -pwm); }
  else              { digitalWrite(L_IN1, LOW);  digitalWrite(L_IN2, LOW);  analogWrite(L_PWM, 0); }
  // R
  if (pwm > 0)      { digitalWrite(R_IN1, HIGH); digitalWrite(R_IN2, LOW);  analogWrite(R_PWM, pwm); }
  else if (pwm < 0) { digitalWrite(R_IN1, LOW);  digitalWrite(R_IN2, HIGH); analogWrite(R_PWM, -pwm); }
  else              { digitalWrite(R_IN1, LOW);  digitalWrite(R_IN2, LOW);  analogWrite(R_PWM, 0); }
}

void steerRelease() { setSteerMotor(0); mode = M_IDLE; }

void loadCal() {
  Calib c; EEPROM.get(0, c);
  if (c.magic == CAL_MAGIC && c.potMax - c.potMin > 40) { cal = c; }
}
void saveCal() { cal.magic = CAL_MAGIC; EEPROM.put(0, cal); }

void printCfg() {
  Serial.print("CFG min="); Serial.print(cal.potMin);
  Serial.print(" max="); Serial.print(cal.potMax);
  Serial.print(" center="); Serial.print(center());
  Serial.print(" incA="); Serial.print(cal.increaseA);
  Serial.print(" drivePwm="); Serial.print(STEER_DRIVE_PWM);
  Serial.print(" deadband="); Serial.println(STEER_DEADBAND);
}

// 스톨(끝단) 판정: 한 창(window)에서 med 변화가 STALL_MIN_DELTA 미만이면 stallWindows++,
// 조금이라도 움직이면 0으로 리셋. STALL_WINDOWS_NEEDED번 "연속" 안 움직여야 스톨 확정.
// 순간적으로 뻑뻑한 지점을 지날 때(한두 창 느려짐)는 스톨로 오판하지 않는다.
bool stalled(int med, unsigned long now) {
  if (now - tStallRef < STALL_WINDOW_MS) return false;
  bool little = abs(med - potAtStallRef) < STALL_MIN_DELTA;
  tStallRef = now; potAtStallRef = med;
  stallWindows = little ? stallWindows + 1 : 0;
  return stallWindows >= STALL_WINDOWS_NEEDED;
}

// 인자를 Mode가 아니라 int로 받는다: Arduino IDE가 함수 프로토타입을 enum Mode 정의보다
// 위에 자동 삽입해 "Mode was not declared" 를 내는 문제를 피하기 위함. (호출부는 enum 값을
// 넘겨도 int로 암시변환됨) 내부에서 Mode로 캐스팅.
void enterPhase(int m, int med) {
  mode = (Mode)m; tPhaseStart = millis(); tStallRef = millis();
  potAtStallRef = med; potPhaseStart = med; stallWindows = 0;
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);
  pinMode(STEER_PWM, OUTPUT); pinMode(STEER_IN1, OUTPUT); pinMode(STEER_IN2, OUTPUT);
  pinMode(L_PWM, OUTPUT); pinMode(L_IN1, OUTPUT); pinMode(L_IN2, OUTPUT);
  pinMode(R_PWM, OUTPUT); pinMode(R_IN1, OUTPUT); pinMode(R_IN2, OUTPUT);
  pinMode(S_POT, INPUT);
  setSteerMotor(0); setDriveMotors(0);

  int v = analogRead(S_POT);
  for (int i = 0; i < 5; i++) medBuf[i] = v;

  loadCal();
  unsigned long now = millis();
  tReport = tStallRef = now;
  Serial.println("READY fw=closed-loop-steer v1");
  printCfg();
}

void parseCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd == "STOP") { setDriveMotors(0); steerRelease(); Serial.println("OK STOP"); return; }
  if (cmd == "SH")   { steerRelease(); Serial.println("OK SH"); return; }
  if (cmd == "SC")   { steerTarget = center(); stalledAtTarget = false; signFlipTried = false;
                       stallWindows = 0; tStallRef = millis(); potAtStallRef = median5();
                       mode = M_HOLD; telemetry = true;
                       Serial.print("OK SC "); Serial.println(steerTarget); return; }
  if (cmd == "CFG")  { printCfg(); return; }
  if (cmd == "FBON") { telemetry = true;  Serial.println("OK FBON"); return; }
  if (cmd == "FBOFF"){ telemetry = false; Serial.println("OK FBOFF"); return; }

  // 진단용: 폐루프/소프트리밋 없이 조향모터를 생짜 PWM으로 구동. +면 극성A(IN1), -면 극성B(IN2).
  // SR 255 / SR -255 로 양쪽 방향 최대토크를 직접 걸어 드라이버 출력전압을 재보는 용도.
  if (cmd.startsWith("SR ")) {
    rawSteer = constrain(cmd.substring(cmd.indexOf(' ') + 1).toInt(), -255, 255);
    mode = M_RAW; telemetry = true;
    Serial.print("OK SR "); Serial.println(rawSteer);
    return;
  }

  if (cmd == "CAL") {
    calEndA = calEndB = -1;
    cal.potMin = 1023; cal.potMax = 0;   // 스윕으로 다시 채운다
    Serial.println("EVENT CAL start (sweep A)");
    enterPhase(M_CAL_A, median5());
    return;
  }

  if (cmd.startsWith("M ")) {
    int throttle = cmd.substring(cmd.indexOf(' ') + 1).toInt();
    throttle = constrain(throttle, -MAX_DRIVE_PWM, MAX_DRIVE_PWM);
    setDriveMotors(throttle);
    telemetry = true;
    Serial.print("OK M "); Serial.println(throttle);
    return;
  }

  if (cmd.startsWith("SA ")) {
    int t = cmd.substring(cmd.indexOf(' ') + 1).toInt();
    t = constrain(t, loLimit(), hiLimit());   // 소프트리밋
    steerTarget = t;
    stalledAtTarget = false; signFlipTried = false; telemetry = true;
    stallWindows = 0; tStallRef = millis(); potAtStallRef = median5();
    // 방향부호 미상이면 먼저 probe, 아니면 바로 hold 제어
    if (cal.increaseA < 0) { Serial.println("EVENT sign probe"); enterPhase(M_PROBE, median5()); potAtProbe = median5(); tProbe = millis(); }
    else                   { mode = M_HOLD; }
    Serial.print("OK SA "); Serial.println(steerTarget);
    return;
  }

  Serial.print("ERR UNKNOWN "); Serial.println(cmd);
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') { parseCommand(inputLine); inputLine = ""; }
    else if (c != '\r') inputLine += c;
  }
}

// "ADC를 올리려면 어느 극성인가"에 따라, 목표로 가는 조향모터 부호를 계산
int steerDirToward(int err) {
  // err = target - pot. err>0 → ADC를 올려야. increaseA=1이면 극성 A(+)가 ADC를 올림.
  bool needIncrease = err > 0;
  bool useA = (needIncrease == (cal.increaseA == 1));
  return useA ? +1 : -1;
}

void loop() {
  unsigned long now = millis();

  medBuf[medIdx] = analogRead(S_POT);
  medIdx = (medIdx + 1) % 5;
  int med = median5();
  if (mode >= M_CAL_A) {                 // 캘리브 중엔 범위 갱신
    if (med < cal.potMin) cal.potMin = med;
    if (med > cal.potMax) cal.potMax = med;
  }

  readSerial();

  switch (mode) {
    case M_IDLE:
      setSteerMotor(0); steerPwmOut = 0;
      break;

    case M_RAW:                                    // 진단: 생짜 구동 (SH/SR 0로 정지)
      setSteerMotor(rawSteer); steerPwmOut = rawSteer;
      break;

    case M_HOLD: {
      // 연속 구동 제어: 목표 방향으로 STEER_DRIVE_PWM으로 계속 밀다가 deadband 안에서 정지.
      // 비백드라이브 기어박스라 모터 끄면 그 자리 유지. 목표 근처 헌팅은 모터 노이즈가 pot
      // 읽기에 타서 생기는 것 -> 근본해결은 pot 와이퍼-GND에 0.1uF 캡(배선 정리).
      if (steerTarget < 0) { setSteerMotor(0); steerPwmOut = 0; break; }
      int err = steerTarget - med;
      if (abs(err) <= STEER_DEADBAND) {              // 도달 -> 정지 유지
        setSteerMotor(0); steerPwmOut = 0; stalledAtTarget = false;
        tStallRef = now; potAtStallRef = med; stallWindows = 0; break;
      }
      if (stalledAtTarget) { setSteerMotor(0); steerPwmOut = 0; break; }  // latch
      if (stalled(med, now)) {                       // 안 움직임 -> 끝단/막힘
        stalledAtTarget = true; setSteerMotor(0); steerPwmOut = 0;
        Serial.print("EVENT STEER STALL pot="); Serial.println(med); break;
      }
      steerPwmOut = steerDirToward(err) * STEER_DRIVE_PWM;
      setSteerMotor(steerPwmOut);
      break;
    }

    case M_PROBE: {
      // 극성 A(+)로 강하게 구동하며 ADC가 오르는지 관찰 → increaseA 확정
      setSteerMotor(STEER_SWEEP_PWM); steerPwmOut = STEER_SWEEP_PWM;
      if (now - tProbe >= PROBE_MS) {
        int d = med - potAtProbe;
        setSteerMotor(0);
        if (abs(d) < STALL_MIN_DELTA) { Serial.println("EVENT probe: no motion (blocked/end) -- retry SA or CAL"); mode = M_IDLE; }
        else { cal.increaseA = (d > 0) ? 1 : 0; saveCal();
               Serial.print("EVENT sign incA="); Serial.println(cal.increaseA);
               mode = M_HOLD; }
      }
      break;
    }

    // ── CAL 자동 스윕: A끝 → B끝 → EEPROM 저장 → 중앙 복귀 ──
    case M_CAL_A:
      setSteerMotor(+calSweepPwm); steerPwmOut = +calSweepPwm;
      // 최소 이동거리 전의 스톨은 false stall(약한 구동/노이즈)로 보고 무시 -> 계속 밀기
      if (now - tPhaseStart > MAX_PHASE_MS ||
          (stalled(med, now) && abs(med - potPhaseStart) >= MIN_SWEEP_TRAVEL)) {
        setSteerMotor(0); tSettle = now; enterPhase(M_CAL_SA, med);
      }
      break;
    case M_CAL_SA:
      setSteerMotor(0); steerPwmOut = 0;
      if (now - tSettle >= SETTLE_MS) {
        calEndA = med; Serial.print("EVENT endA="); Serial.println(calEndA);
        enterPhase(M_CAL_B, med);
      }
      break;
    case M_CAL_B:
      setSteerMotor(-calSweepPwm); steerPwmOut = -calSweepPwm;
      if (now - tPhaseStart > MAX_PHASE_MS ||
          (stalled(med, now) && abs(med - potPhaseStart) >= MIN_SWEEP_TRAVEL)) {
        setSteerMotor(0); tSettle = now; enterPhase(M_CAL_SB, med);
      }
      break;
    case M_CAL_SB:
      setSteerMotor(0); steerPwmOut = 0;
      if (now - tSettle >= SETTLE_MS) {
        calEndB = med; Serial.print("EVENT endB="); Serial.println(calEndB);
        cal.potMin = min(calEndA, calEndB);
        cal.potMax = max(calEndA, calEndB);
        cal.increaseA = (calEndA > calEndB) ? 1 : 0;   // 극성 A가 더 큰 ADC면 A가 올림
        saveCal();
        Serial.print("SUMMARY min="); Serial.print(cal.potMin);
        Serial.print(" max="); Serial.print(cal.potMax);
        Serial.print(" span="); Serial.print(cal.potMax - cal.potMin);
        Serial.print(" center="); Serial.print(center());
        Serial.print(" incA="); Serial.println(cal.increaseA);
        steerTarget = center();
        signFlipTried = false;
        enterPhase(M_CAL_CENTER, med);
      }
      break;
    case M_CAL_CENTER: {
      int err = steerTarget - med;
      if (abs(err) <= STEER_DEADBAND || now - tPhaseStart > MAX_PHASE_MS) {
        setSteerMotor(0); steerPwmOut = 0; mode = M_HOLD;
        Serial.print("EVENT CAL done, centered pot="); Serial.println(med);
        telemetry = false;   // FB 스트림 정지 -> SUMMARY가 화면에 남음 (재개: FBON 또는 구동명령)
        Serial.println("EVENT telemetry off (send FBON or any drive cmd to resume)");
      } else {
        steerPwmOut = steerDirToward(err) * calSweepPwm; setSteerMotor(steerPwmOut);
      }
      break;
    }
  }

  // FB 스트림: telemetry가 켜져 있고 CAL 중이 아닐 때만 (CAL 중엔 EVENT/SUMMARY만 깔끔히 보이게)
  if (now - tReport >= REPORT_MS) {
    tReport = now;
    if (telemetry && mode < M_CAL_A) {
      Serial.print("FB "); Serial.print(med); Serial.print(' ');
      Serial.print(steerTarget); Serial.print(' ');
      Serial.print(steerPwmOut); Serial.print(' ');
      Serial.println(mode == M_HOLD ? 1 : 0);
    }
  }
}
