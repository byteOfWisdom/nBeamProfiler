#define SET_PIN_MODE_OUTPUT(port, pin) DDR ## port |= (1 << pin)
#define SET_PIN_HIGH(port, pin) (PORT ## port |= (1 << pin))
#define SET_PIN_LOW(port, pin) ((PORT ## port) &= ~(1 << (pin)))

#define INT_PIN 2

inline void rect_pulse(void) {
  SET_PIN_HIGH(B, 0);  
  delay(10);
  SET_PIN_LOW(B, 0);
}

void setup() {
  SET_PIN_MODE_OUTPUT(B, 0);
  pinMode(INT_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(INT_PIN), rect_pulse, CHANGE);
}

void loop() {
  delay(1000);
}
