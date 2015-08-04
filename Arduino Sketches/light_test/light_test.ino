int leds[] = {13, 12, 11};

void setup () {
  for (int i=11;i<=13;i++) {
    pinMode(i, OUTPUT); 
  }
}

void loop () {
  for (int i=11;i<=13;i++) {
    digitalWrite(i, HIGH);
    delay(400);
    digitalWrite(i, LOW); 
  }
}
