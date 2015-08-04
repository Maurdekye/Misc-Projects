int led = 13;
int button = 12;

void setup () {
  pinMode(led, OUTPUT);
  pinMode(button, INPUT);
}

void loop () {
  if (digitalRead(button)) {
   digitalWrite(led, HIGH);
  }
  delay(150);
  digitalWrite(led, LOW);
  delay(150);
}
