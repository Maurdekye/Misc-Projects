int leds[] = {11, 12, 13};
int up = 8;
int down = 9;
boolean upPressed = false;
boolean downPressed = false;
int curLed = 0;

void setup () {
  pinMode(up, INPUT);
  pinMode(down, OUTPUT);
  for (int i=leds[0];i<=leds[2];i++) {
    pinMode(i, OUTPUT);
  }
}

void loop () {
  for (int i=0;i<=2;i++) {
    if (leds[i] == leds[curLed]) {
      digitalWrite(leds[i], HIGH); 
    } else {
      digitalWrite(leds[i], LOW); 
    }
  }
  int cycleUp = digitalRead(up);
  int cycleDown = digitalRead(down);
  if (cycleUp != upPressed && !cycleDown && cycleUp) {
    curLed = (curLed + 1)%3;
  } else if (cycleDown != downPressed && !cycleUp && cycleDown) {
    curLed = (curLed - 1)%3;
  }
  upPressed = cycleUp;
  downPressed = cycleDown;
}
