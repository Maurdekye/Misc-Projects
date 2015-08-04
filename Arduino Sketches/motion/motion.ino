int sensor = 2;
int led = 13;

void setup () {
  pinMode(led, OUTPUT);
  pinMode(sensor, INPUT); 
}

void loop () {
  if (digitalRead(sensor)) {
    digitalWrite(led, HIGH);
    delay(50);
    digitalWrite(led, LOW);
    delay(80);
  } else {
    digitalWrite(led, LOW);
  }
}
