int red = 13;
int blue = 12;
int green = 11;
int lightsensor = A0;
int motsensor = 10;

void setup() {
  pinMode(13, OUTPUT);
  pinMode(12, OUTPUT);
  pinMode(11, OUTPUT);
  pinMode(10, INPUT);
}

void loop() {
  if (digitalRead(motsensor)) {
    delay(300);
  }
  digitalWrite(red, LOW);
  digitalWrite(blue, LOW);
  digitalWrite(green, LOW);
  if (digitalRead(motsensor)) {
    delay(300);
  }
  int light = analogRead(lightsensor);
  if (light > 255) {
    digitalWrite(red, HIGH);
  }
  if (light > 511) {
    digitalWrite(blue, HIGH);
  }
  if (light > 767) {
    digitalWrite(green, HIGH);
  }
}
