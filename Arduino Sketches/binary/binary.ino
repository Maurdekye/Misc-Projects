int leds[] = {8, 9, 10, 11, 12, 13};
int add = 1;
int sub = 2;

boolean add_presscheck = false;
boolean sub_presscheck = false;
int length = sizeof(leds);
boolean activated[] = {false, false, false, false, false, false};

void setup () {
  for (int i=0;i<length;i++) {
    pinMode(leds[i], OUTPUT);
  }
  pinMode(add, INPUT);
  pinMode(sub, INPUT);
}

void loop () {
  boolean add_val = digitalRead(add);
  boolean sub_val = digitalRead(sub);
  
  if (add_presscheck != add_val && add_val && !sub_val) {
    for (int i=0;i<length;i++) {
      if (!activated[i]) {
        activated[i] = true;
        break;
      }
      else {
        activated[i] = false;
      }
    }
  } else if (sub_presscheck != sub_val && sub_val && !add_val) {
    for (int i=0;i<length;i++) {
      if (activated[i]) {
        activated[i] = false;
        break;
      }
      else {
        activated[i] = true;
      }
    }
  }
  
  for (int i=0;i<=length;i++) {
    if (activated[i]) {
      digitalWrite(leds[i], HIGH); 
    } else {
      digitalWrite(leds[i], LOW);  
    }
  }
  
  add_presscheck = add_val;
  sub_presscheck = sub_val;
}
