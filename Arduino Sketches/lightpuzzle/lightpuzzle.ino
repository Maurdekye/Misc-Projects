boolean activated[] = {false, true, false, true, true, false};
boolean presstrack[] = {false, false, false, false, false, false};

void setup () {
  for (int i=1;i<=6;i++) {
    pinMode(i, OUTPUT);
    pinMode(i+6, INPUT);
  }
  pinMode(13, OUTPUT);
}

void loop () {
  for (int i=1;i<=6;i++) {
  
    boolean curbutton = digitalRead(i+6);
    if ((presstrack[i-1] != curbutton) && curbutton) {
      activated[i-1] = !activated[i-1];
      if (i < 6) {
        activated[i] = !activated[i];
      }
      if (i > 1) {
        activated[i-2] = !activated[i-2];
      }
    }
    presstrack[i-1] = curbutton;
    
    boolean allactive = true;
    if (activated[i-1]) {
      digitalWrite(i, HIGH);
    } else {
      digitalWrite(i, LOW);
      allactive = false;
    }
    
    if (allactive) {
      delay(1600);
      digitalWrite(13, HIGH);
      for (int j=0;j<4;j++) {
        for (int e=1;e<=6;e++) {
          digitalWrite(e, HIGH);
          delay(400);
          digitalWrite(e, LOW);
        }
      }
      digitalWrite(13, LOW);
      boolean activated[] = {false, true, false, true, true, false};
    }
    
  }
}
