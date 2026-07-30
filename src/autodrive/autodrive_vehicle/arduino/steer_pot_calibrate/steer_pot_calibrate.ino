/*
 * steer_pot_calibrate.ino — 조향을 좌/우 기계 끝단까지 자동으로 쓸어(sweep)
 *   POT(A6)의 min/max/중앙을 뽑는 "재실행 가능한" 캘리브레이션 스케치.
 *
 * 왜 별도 파일인가:
 *   끝단 저항값은 물리적 이슈(링키지 유격, POT 미끄러짐, 재장착)로 드리프트한다.
 *   그래서 수동 l/r(steer_pot_test.ino)로 매번 손으로 끝까지 미는 대신, 명령 한 번에
 *   자동으로 양 끝을 재측정하고 요약을 뱉는 도구가 필요하다. 폐루프 매핑을 갱신할 때마다
 *   이 스케치를 다시 올려 돌리면 된다.
 *
 * ⚠️ 진단/캘리브레이션 전용 — 폐루프 제어 아님(목표각 추종 없음).
 *    구동(주행) 모터는 부팅 시 끄고 이후 절대 건드리지 않는다 — 차는 안 움직인다.
 *
 * 끝단 검출 방식:
 *   기계 끝단에 닿으면 POT 값이 더 이상 변하지 않는다 → 스톨 감지(STALL_WINDOW_MS 동안
 *   med5 변화 < STALL_MIN_DELTA)로 "끝에 도달"을 판정하고 즉시 정지·기록한다.
 *   (POT이 죽어 있어도 스톨로 걸려 안전. 단 그 경우 span이 0에 가까워 경고를 낸다.)
 *
 * ── 시리얼 명령 (115200) ─────────────────────────────────────────
 *   g   자동 캘리브레이션 시작 (DIR_A 끝 → DIR_B 끝 → 중앙 복귀 → 요약)
 *   s   즉시 중단/정지 (언제든)
 *   +/- 스윕 PWM 5씩 증감           p<n> 스윕 PWM 직접 지정 (예: p90)
 *   c   마지막 요약 다시 출력
 *
 *   ※ DIR_A/DIR_B는 드라이버 극성일 뿐 물리 좌/우가 아니다. 어느 쪽이 실제 좌회전인지는
 *     요약의 endA/endB와 바퀴를 눈으로 보고 확정할 것.
 *
 * ── 출력 (매 20ms, 50Hz) ─────────────────────────────────────────
 *   POT <raw> <med5> <min> <max> <phase> <pwm>
 *   EVENT ...            단계 전이/끝단 기록/경고
 *   SUMMARY min=.. max=.. span=.. mid=.. endA=.. endB=..   최종 결과
 */

// ── 핀 (steer_pot_test.ino와 동일, 사용자 확정 배선) ──
const int S_PWM = 2;
const int S_IN1 = 22;
const int S_IN2 = 23;
const int S_POT = A6;

// 구동 모터 — 부팅 시 끄기만 하고 이후 사용하지 않음
const int R_PWM = 3, R_IN1 = 24, R_IN2 = 25;
const int L_PWM = 4, L_IN1 = 26, L_IN2 = 27;

// ── 파라미터 ──
int steerPwm              = 120;   // 스윕 PWM. +/- 또는 p<n> 으로 조정
const int PWM_MIN         = 40;
const int PWM_MAX         = 200;

const unsigned long STALL_WINDOW_MS = 600;   // 이 시간 동안
const int           STALL_MIN_DELTA = 3;     // 이만큼도 안 변하면 끝단(스톨)
const unsigned long MAX_PHASE_MS    = 6000;  // 한 단계 최대 구동(안전 상한)
const unsigned long SETTLE_MS       = 400;   // 끝단 정지 후 값 안정화 대기
const unsigned long REPORT_MS       = 20;    // 50Hz 보고
const int           CENTER_DEADBAND = 8;     // 중앙 복귀 허용 오차(카운트)
const unsigned long CENTER_TIMEOUT_MS = 4000;

// ── 상태 ──
enum Phase { P_IDLE, P_SWEEP_A, P_SETTLE_A, P_SWEEP_B, P_SETTLE_B, P_CENTER, P_DONE, P_ABORT };
Phase phase = P_IDLE;

unsigned long tPhaseStart = 0, tReport = 0, tStallRef = 0, tSettle = 0;
int potAtStallRef = 0;
int potMin = 1023, potMax = 0;
int endA = -1, endB = -1;        // DIR_A / DIR_B 끝단 ADC
int centerTarget = -1;

int medBuf[5] = {0, 0, 0, 0, 0};
int medIdx = 0;

int median5() {
  int a[5];
  for (int i = 0; i < 5; i++) a[i] = medBuf[i];
  for (int i = 1; i < 5; i++) {
    int k = a[i], j = i - 1;
    while (j >= 0 && a[j] > k) { a[j + 1] = a[j]; j--; }
    a[j + 1] = k;
  }
  return a[2];
}

void steerStop() {
  analogWrite(S_PWM, 0);
  digitalWrite(S_IN1, LOW);
  digitalWrite(S_IN2, LOW);
}

void steerA(int pwm) {           // S_IN1 HIGH  (= DIR_A)
  digitalWrite(S_IN1, HIGH);
  digitalWrite(S_IN2, LOW);
  analogWrite(S_PWM, pwm);
}

void steerB(int pwm) {           // S_IN2 HIGH  (= DIR_B)
  digitalWrite(S_IN1, LOW);
  digitalWrite(S_IN2, HIGH);
  analogWrite(S_PWM, pwm);
}

void driveMotorsOff() {
  analogWrite(R_PWM, 0); analogWrite(L_PWM, 0);
  digitalWrite(R_IN1, LOW); digitalWrite(R_IN2, LOW);
  digitalWrite(L_IN1, LOW); digitalWrite(L_IN2, LOW);
}

void enterPhase(Phase p) {
  phase = p;
  tPhaseStart = millis();
  tStallRef = millis();
  potAtStallRef = median5();
}

void printSummary() {
  int span = (endA >= 0 && endB >= 0) ? abs(endA - endB) : (potMax - potMin);
  Serial.print("SUMMARY min="); Serial.print(potMin);
  Serial.print(" max="); Serial.print(potMax);
  Serial.print(" span="); Serial.print(span);
  Serial.print(" mid="); Serial.print((potMin + potMax) / 2);
  Serial.print(" endA="); Serial.print(endA);
  Serial.print(" endB="); Serial.println(endB);
  if (span < 40) {
    Serial.println("EVENT WARN span too small -- POT may be dead/uncoupled or PWM too low");
  }
}

// 스톨(끝단) 도달 여부: STALL_WINDOW_MS 동안 med 변화가 STALL_MIN_DELTA 미만.
// true 반환 시 caller가 끝단으로 확정한다. 창(window)마다 기준값을 갱신한다.
bool stalled(int med, unsigned long now) {
  if (now - tStallRef < STALL_WINDOW_MS) return false;
  bool s = abs(med - potAtStallRef) < STALL_MIN_DELTA;
  tStallRef = now;
  potAtStallRef = med;
  return s;
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);

  pinMode(S_PWM, OUTPUT); pinMode(S_IN1, OUTPUT); pinMode(S_IN2, OUTPUT);
  pinMode(R_PWM, OUTPUT); pinMode(R_IN1, OUTPUT); pinMode(R_IN2, OUTPUT);
  pinMode(L_PWM, OUTPUT); pinMode(L_IN1, OUTPUT); pinMode(L_IN2, OUTPUT);
  pinMode(S_POT, INPUT);

  steerStop();
  driveMotorsOff();

  int v = analogRead(S_POT);
  for (int i = 0; i < 5; i++) medBuf[i] = v;
  potMin = potMax = v;
  potAtStallRef = v;

  unsigned long now = millis();
  tPhaseStart = tReport = tStallRef = now;

  Serial.println("STEERPOT CALIBRATE READY  cmds: g(start) s(stop) +/- p<n> c");
}

void handleSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'g' || c == 'G') {
      // 새 캘리브레이션 시작 — 범위/끝단 초기화 후 DIR_A 스윕
      potMin = potMax = median5();
      endA = endB = -1;
      Serial.println("EVENT start sweep DIR_A");
      enterPhase(P_SWEEP_A);
    } else if (c == 's' || c == 'S' || c == '0') {
      steerStop();
      phase = P_ABORT;
      Serial.println("EVENT abort (user stop)");
    } else if (c == '+') {
      steerPwm = min(PWM_MAX, steerPwm + 5);
      Serial.print("PWM "); Serial.println(steerPwm);
    } else if (c == '-') {
      steerPwm = max(PWM_MIN, steerPwm - 5);
      Serial.print("PWM "); Serial.println(steerPwm);
    } else if (c == 'p' || c == 'P') {
      long v = Serial.parseInt();
      if (v > 0) { steerPwm = constrain((int)v, PWM_MIN, PWM_MAX);
                   Serial.print("PWM "); Serial.println(steerPwm); }
    } else if (c == 'c' || c == 'C') {
      printSummary();
    }
  }
}

void loop() {
  unsigned long now = millis();

  // POT 샘플링 (매 루프)
  medBuf[medIdx] = analogRead(S_POT);
  medIdx = (medIdx + 1) % 5;
  int med = median5();
  if (med < potMin) potMin = med;
  if (med > potMax) potMax = med;

  handleSerial();

  // ── 단계별 로직 ──
  switch (phase) {
    case P_SWEEP_A:
      if (now - tPhaseStart > MAX_PHASE_MS) {          // 안전 상한
        steerStop(); endA = med; tSettle = now;
        Serial.print("EVENT endA(timeout)="); Serial.println(endA);
        enterPhase(P_SETTLE_A);
      } else if (stalled(med, now)) {                   // 끝단 도달
        steerStop(); tSettle = now;
        Serial.println("EVENT DIR_A end reached");
        enterPhase(P_SETTLE_A);
      }
      break;

    case P_SETTLE_A:
      if (now - tSettle >= SETTLE_MS) {                 // 값 안정화 후 기록
        endA = med;
        Serial.print("EVENT endA="); Serial.println(endA);
        Serial.println("EVENT start sweep DIR_B");
        enterPhase(P_SWEEP_B);
      }
      break;

    case P_SWEEP_B:
      if (now - tPhaseStart > MAX_PHASE_MS) {
        steerStop(); tSettle = now;
        Serial.println("EVENT DIR_B stop (timeout)");
        enterPhase(P_SETTLE_B);
      } else if (stalled(med, now)) {
        steerStop(); tSettle = now;
        Serial.println("EVENT DIR_B end reached");
        enterPhase(P_SETTLE_B);
      }
      break;

    case P_SETTLE_B:
      if (now - tSettle >= SETTLE_MS) {
        endB = med;
        Serial.print("EVENT endB="); Serial.println(endB);
        printSummary();
        // 중앙 복귀 준비 (측정 끝단으로 방향 부호 결정)
        centerTarget = (potMin + potMax) / 2;
        Serial.print("EVENT centering to "); Serial.println(centerTarget);
        enterPhase(P_CENTER);
      }
      break;

    case P_CENTER: {
      // 중앙 목표로 복귀: endA/endB로 "med를 줄이는 방향"을 안다.
      //   endA < endB 이면 DIR_A가 med를 줄인다 (그 반대면 DIR_B).
      if (now - tPhaseStart > CENTER_TIMEOUT_MS) {
        steerStop(); phase = P_DONE;
        Serial.println("EVENT center timeout -- stopped");
        break;
      }
      int err = med - centerTarget;                     // >0: 너무 큼 → 줄여야
      if (abs(err) <= CENTER_DEADBAND) {
        steerStop(); phase = P_DONE;
        Serial.print("EVENT centered pot="); Serial.println(med);
        break;
      }
      bool dirA_decreases = (endA <= endB);
      bool needDecrease = err > 0;
      bool useA = (needDecrease == dirA_decreases);
      if (useA) steerA(steerPwm); else steerB(steerPwm);
      break;
    }

    case P_IDLE:
    case P_DONE:
    case P_ABORT:
    default:
      steerStop();
      break;
  }

  // 스윕 중 모터 출력 유지 (P_CENTER는 위에서 직접 구동)
  if (phase == P_SWEEP_A) steerA(steerPwm);
  else if (phase == P_SWEEP_B) steerB(steerPwm);

  // ── 보고 ──
  if (now - tReport >= REPORT_MS) {
    tReport = now;
    const char* pn =
      phase == P_IDLE     ? "IDLE"   :
      phase == P_SWEEP_A  ? "SWEEP_A":
      phase == P_SETTLE_A ? "SETTLE_A":
      phase == P_SWEEP_B  ? "SWEEP_B":
      phase == P_SETTLE_B ? "SETTLE_B":
      phase == P_CENTER   ? "CENTER" :
      phase == P_DONE     ? "DONE"   : "ABORT";
    bool driving = (phase == P_SWEEP_A || phase == P_SWEEP_B || phase == P_CENTER);
    Serial.print("POT "); Serial.print(analogRead(S_POT)); Serial.print(' ');
    Serial.print(med); Serial.print(' ');
    Serial.print(potMin); Serial.print(' '); Serial.print(potMax); Serial.print(' ');
    Serial.print(pn); Serial.print(' ');
    Serial.println(driving ? steerPwm : 0);
  }
}
