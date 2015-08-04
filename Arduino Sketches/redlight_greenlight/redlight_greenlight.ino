int red = 1;
int green = 2;
int blue = 3;
int sensor = 13;

void setup() {
  pinMode(red, OUTPUT);
  pinMode(green, OUTPUT);
  pinMode(blue, OUTPUT);
  pinMode(sensor, INPUT);
}

void loop() {
  setColor(false, true, false);
  delay(12000);
  setColor(true, true, false);
  delay(8000);
  unsigned long clock = millis();
  boolean seen = false;
  setColor(true, false, false);
  while (millis() - 5000 < clock) {
     if (digitalRead(sensor)) {
       seen = true;
       break;
     }
  }
  if (seen) {
    boolean onoff = true;
    for (int i=0;i<16;i++) {
      setColor(onoff, false, false);
      onoff = !onoff;
      delay(250);
    }
  }
}

void setColor(int r, int g, int b) {
  digitalWrite(red, r);
  digitalWrite(green, g);
  digitalWrite(blue, b);
}
