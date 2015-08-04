int A = 8;
int B = 9;
int AND_LED = 11;
int OR_LED = 12;
int XOR_LED = 13;

void setup () {
  pinMode(A, INPUT);
  pinMode(B, INPUT);
  pinMode(AND_LED, OUTPUT);
  pinMode(OR_LED, OUTPUT);
  pinMode(XOR_LED, OUTPUT);
}

void loop () {
  if (digitalRead(A) && digitalRead(B)) {
    digitalWrite(AND_LED, HIGH);
  } else {
    digitalWrite(AND_LED, LOW);
  }
  if (digitalRead(A) || digitalRead(B)) {
    digitalWrite(OR_LED, HIGH);
  } else {
    digitalWrite(OR_LED, LOW);
  }
  
  if (digitalRead(A) != digitalRead(B)) {
    digitalWrite(XOR_LED, HIGH);
  } else {
    digitalWrite(XOR_LED, LOW);
  }
}
